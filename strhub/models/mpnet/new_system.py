import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import math
import numpy as np
from functools import partial
from itertools import permutations
from typing import Sequence, Any, Optional, Tuple, List

from pytorch_lightning.utilities.types import STEP_OUTPUT
from timm.models.helpers import named_apply

from strhub.data.utils import MPTokenizer
from strhub.models.base import BaseSystem
from strhub.models.utils import init_weights
from .modules import Encoder, TokenEmbedding, Decoder, DecoderLayer, PadPrediction


class MPNet(BaseSystem):
    def __init__(self, charset_train: str, charset_test: str, max_label_length: int,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float,
                 img_size: Sequence[int], patch_size: Sequence[int], embed_dim: int,
                 enc_num_heads: int, enc_mlp_ratio: int, enc_depth: int,
                 dec_num_heads: int, dec_mlp_ratio: int, dec_depth: int,
                 perm_num: int, perm_forward: bool, perm_mirrored: bool,
                 decode_ar: bool, dropout: float, len_token: bool,
                 len_ratio: float, use_gt: bool, refine_iters: int,
                 top_k: int, mlm_iters: int, **kwargs: Any) -> None:
        tokenizer = MPTokenizer(charset_train)
        super().__init__(tokenizer, charset_test, batch_size, lr, warmup_pct, weight_decay)
        self.bos_id = tokenizer.bos_id
        self.eos_id = tokenizer.eos_id
        self.pad_id = tokenizer.pad_id
        self.mask_id = tokenizer.mask_id

        self.save_hyperparameters()

        self.len_token = len_token
        self.max_label_length = max_label_length
        self.decode_ar = decode_ar
        self.use_gt = use_gt
        self.refine_iters = refine_iters
        self.top_k = top_k
        self.mlm_iter = math.ceil((max_label_length + 1) / self.top_k)
        self.len_ratio = len_ratio
        self.detach = False

        self.rng = np.random.default_rng()
        self.max_gen_perms = perm_num // 2 if perm_mirrored else perm_num
        self.perm_forward = perm_forward
        self.perm_mirrored = perm_mirrored

        # We don't predict <mask> nor <bos>
        self.head = nn.Linear(embed_dim, len(self.tokenizer) - 2)
        self.text_embed = TokenEmbedding(len(self.tokenizer), embed_dim)

        # +1 for <eos>
        self.pos_queries = nn.Parameter(torch.Tensor(1, max_label_length + 1, embed_dim))
        self.dropout = nn.Dropout(p=dropout)

        if len_token:
            self.encoder = Encoder(img_size, patch_size, embed_dim=embed_dim, depth=enc_depth,
                                   num_heads=enc_num_heads, mlp_ratio=enc_mlp_ratio,
                                   class_token=True)
            # predict the rate of mask for key_padding_mask
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, 128),
                nn.Linear(128, self.max_label_length + 2)  # +2 for the <bos> and <eos> token
            )
        else:
            self.encoder = Encoder(img_size, patch_size, embed_dim=embed_dim, depth=enc_depth,
                                   num_heads=enc_num_heads, mlp_ratio=enc_mlp_ratio,
                                   class_token=False)

        self.pad_predict = PadPrediction(embed_dim, dec_num_heads, embed_dim * dec_mlp_ratio, dropout)
        decoder_layer = DecoderLayer(embed_dim, dec_num_heads, embed_dim * dec_mlp_ratio, dropout)
        self.decoder = Decoder(decoder_layer, num_layers=dec_depth, norm=nn.LayerNorm(embed_dim))

        # Encoder has its own init.
        named_apply(partial(init_weights, exclude=['encoder']), self)
        nn.init.trunc_normal_(self.pos_queries, std=.02)

    def encode(self, img: torch.Tensor):
        return self.encoder(img)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[Tensor] = None,
               tgt_padding_mask: Optional[Tensor] = None, tgt_query: Optional[Tensor] = None,
               tgt_query_mask: Optional[Tensor] = None, max_length: Optional[int] = None,
               tgt_extend: Optional[Tensor] = None):
        """
        While training, L = 2 * (T+2)
        """
        N, L = tgt.shape
        if tgt_query is None:
            L_tgt = L // 2  # (T+2)
            # <bos> stands for the null context.
            null_ctx = self.text_embed(tgt[:, :1])
            tgt_emb = self.pos_queries[:, :L_tgt - 1] + self.text_embed(tgt[:, 1:L_tgt])
            mask_null_ctx = self.text_embed(tgt[:, L_tgt:L_tgt + 1])
            pos_emb = self.pos_queries[:, :L_tgt - 1] + self.text_embed(tgt[:, L_tgt + 1:])
            tgt_emb = self.dropout(torch.cat([null_ctx, tgt_emb, mask_null_ctx, pos_emb], dim=1))
            tgt_query = self.pos_queries[:, :L_tgt - 1].expand(N, -1, -1)
        else:
            # <bos> stands for the null context. We only supply position information for characters after <bos>.
            if max_length is None:
                max_length = self.max_label_length + 1
            null_ctx = self.text_embed(tgt[:, :1])
            tgt_emb = self.pos_queries[:, :L - 1] + self.text_embed(tgt[:, 1:])
            mask_null_ctx = self.text_embed(torch.full((N, 1), self.mask_id, dtype=torch.long, device=self._device))
            if tgt_extend is None:
                # use the masks
                pos_emb = self.pos_queries[:, :max_length] + \
                    self.text_embed(torch.full((N, max_length), self.mask_id, dtype=torch.long, device=self._device))
            else:
                # use the pad_id for masked values
                pos_emb = self.pos_queries[:, :max_length] + self.text_embed(tgt_extend[:, :max_length])
            tgt_emb = self.dropout(torch.cat([null_ctx, tgt_emb, mask_null_ctx, pos_emb], dim=1))
            # tgt_emb = self.dropout(torch.cat([null_ctx, tgt_emb], dim=1))

        tgt_query = self.dropout(tgt_query)
        return self.decoder(tgt_query, tgt_emb, memory, tgt_query_mask, tgt_mask, tgt_padding_mask)

    def iterative_decode(self, feature_map: torch.Tensor, target: Optional[Tensor] = None,
                         predict_len: Optional[Tensor] = None):
        """
        Easy first decoding strategy
        :param feature_map:
        :param input_labels
        """
        # torch.autograd.set_detect_anomaly(True)
        bs = feature_map.shape[0]  # the batch size
        C = feature_map.shape[-1]

        tgt_tokens = torch.full((bs, self.max_label_length + 1), self.mask_id, dtype=torch.long, device=self._device)
        pred_tgt_tokens = torch.full((bs, self.max_label_length + 1), self.pad_id, dtype=torch.long,
                                     device=self._device)
        token_logits = torch.zeros([bs, self.max_label_length + 1, len(self.tokenizer) - 2], dtype=torch.float,
                                   device=self._device)
        final_ffn = torch.zeros([bs, self.max_label_length + 1, C], dtype=torch.float, device=self._device)

        if target is not None:  # is training
            tgt_padding_mask = torch.cat([torch.full((bs, 1), False, dtype=torch.bool, device=self._device),
                                          (target == self.pad_id)], dim=1)
        else:
            tgt_padding_mask = torch.full((bs, self.max_label_length + 2), False, dtype=torch.bool, device=self._device)

        # Query all positions up to `max_label_length + 1`
        bos_token = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=self._device)
        null_ctx = self.text_embed(bos_token)
        tgt_query = self.pos_queries[:, :self.max_label_length + 1].expand(bs, -1, -1)

        for i in range(self.mlm_iter):
            # <bos> stands for the null context.
            tgt_emb = self.pos_queries[:, :self.max_label_length + 1] + self.text_embed(tgt_tokens)
            tgt_emb = self.dropout(torch.cat([null_ctx, tgt_emb], dim=1))
            tgt_out = self.decoder(tgt_query, tgt_emb, feature_map, content_key_padding_mask=tgt_padding_mask)

            new_token_logits = self.head(tgt_out)
            token_probs = new_token_logits.softmax(-1)
            new_ffn = tgt_out

            new_tgt_tokens = torch.argmax(new_token_logits, dim=-1)
            # only predict the mask ones
            token_probs = torch.max(token_probs, dim=-1).values  # N*T
            token_probs = torch.where(tgt_tokens == self.mask_id, token_probs, torch.zeros_like(token_probs))
            '''
            probs_mask = torch.full(token_probs.shape, float('-inf'), device=self._device)
            if target is not None:
                token_probs = torch.where(target==self.pad_id, probs_mask, token_probs)
            else:
                for i in range(len(predict_len)):
                    token_probs[i, predict_len[i]+1:] = probs_mask[i, predict_len[i]+1:]             
            '''
            top_tuple = token_probs.topk(self.top_k, dim=1)  # get the top-k best position
            kth = torch.min(top_tuple.values, dim=1, keepdim=True).values
            update_idx = torch.greater_equal(token_probs, kth)

            logits_update_idx = torch.tile(update_idx.unsqueeze(dim=2), [1, 1, len(self.tokenizer) - 2])
            ffn_update_idx = torch.tile(update_idx.unsqueeze(dim=2), [1, 1, C])

            if target is not None:  # is training
                tgt_tokens = torch.where(update_idx, target, tgt_tokens)
                pred_tgt_tokens = torch.where(update_idx, new_tgt_tokens, pred_tgt_tokens)

            else:  # testing
                tgt_tokens = torch.where(update_idx, new_tgt_tokens, tgt_tokens)
                pred_tgt_tokens = tgt_tokens

            token_logits = torch.where(logits_update_idx, new_token_logits, token_logits)
            final_ffn = torch.where(ffn_update_idx, new_ffn, final_ffn)

        return token_logits, pred_tgt_tokens, final_ffn

    def forward(self, images: Tensor, max_length: Optional[int] = None, target: Optional[Tensor] = None) -> Tensor:
        memory = self.encode(images)
        bs = images.shape[0]
        pad_query = self.pos_queries[:, :self.max_label_length + 1].expand(bs, -1, -1)
        if self.len_token:
            # predict the labels' length using the [len] token
            # padding_mask = padding_pred = self.mlp_head(memory[:, 0])  # (bs, self.max_label_length + 2)
            # print(memory[:, 0].shape)
            pad_query = pad_query + memory[:, 0].unsqueeze(1).expand(-1, self.max_label_length + 1, -1)
        padding_pred = self.pad_predict(pad_query, memory[:, 1:] if self.len_token else memory)
        softmax_padding_pred = F.softmax(padding_pred, dim=2)
        padding_mask = torch.cat([torch.full((bs, 1), 0.0, device=self._device), softmax_padding_pred[:, :, 1]], dim=1)
        # print(padding_pred)
        # print(padding_mask)
        testing = max_length is None
        max_length = self.max_label_length if max_length is None else min(max_length, self.max_label_length)
        # +1 for <eos> at end of sequence.
        num_steps = max_length + 1
        # Query positions up to `num_steps`
        tgt_query = self.pos_queries[:, :num_steps].expand(bs, -1, -1)

        tgt_padding_mask = torch.full((bs, num_steps), 0.0, dtype=torch.float, device=self._device)
        padding_mask = padding_mask[:, :num_steps + 1]
        tgt_extend = torch.full((bs, num_steps+1), self.mask_id, dtype=torch.long, device=self._device)

        # print(padding_mask)
        if target is not None:  # if use the gt
            padding_mask = torch.full(target.shape, 0.0, dtype=torch.float, device=self._device)
            padding_mask = torch.where(target == self.pad_id, 1.0, padding_mask)
            tgt_extend = torch.where(target == self.pad_id, self.pad_id, tgt_extend)
        else:
            tgt_extend = torch.where(padding_mask > 0.99, self.pad_id, tgt_extend)

        # Special case for the forward permutation. Faster than using `generate_attn_masks()`
        tgt_mask = query_mask = torch.triu(torch.full((num_steps, num_steps), float('-inf'), device=self._device), 1)
        query_mask_extend = torch.tril(torch.full((num_steps, num_steps + 1), float('-inf'), device=self._device), 0)

        if self.decode_ar:
            tgt_in = torch.full((bs, num_steps), self.pad_id, dtype=torch.long, device=self._device)
            tgt_in[:, 0] = self.bos_id

            logits = []
            for i in range(num_steps):
                j = i + 1  # next token index
                # Efficient decoding:
                # Input the context up to the ith token. We use only one query (at position = i) at a time.
                # This works because of the lookahead masking effect of the canonical (forward) AR context.
                # Past tokens have no access to future tokens, hence are fixed once computed.
                tgt_out = self.decode(tgt_in[:, :j], memory[:, 1:] if self.len_token else memory, tgt_mask[:j, :j],
                                      tgt_padding_mask=torch.concat([tgt_padding_mask[:, :j], padding_mask], dim=-1),
                                      tgt_query=tgt_query[:, i:j],
                                      tgt_query_mask=torch.concat([query_mask[i:j, :j], query_mask_extend[i:j, :]],
                                                                  dim=-1), max_length=num_steps,
                                      tgt_extend=tgt_extend)
                # the next token probability is in the output's ith token position
                p_i = self.head(tgt_out)
                logits.append(p_i)
                if j < num_steps:
                    # greedy decode. add the next token index to the target input
                    tgt_in[:, j] = p_i.squeeze().argmax(-1)
                    # Efficient batch decoding: If all output words have at least one EOS token, end decoding.
                    if testing and (tgt_in == self.eos_id).any(dim=-1).all():
                        break

            logits = torch.cat(logits, dim=1)
        else:
            # No prior context, so input is just <bos>. We query all positions.
            tgt_in = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=self._device)
            tgt_out = self.decode(tgt_in, memory[:, 1:] if self.len_token else memory,
                                  tgt_padding_mask=torch.concat([tgt_padding_mask[:, :1], padding_mask], dim=-1),
                                  tgt_query=tgt_query,
                                  tgt_extend=tgt_extend)
            logits = self.head(tgt_out)
            # print(logits[0])
            # logits, _, _ = self.iterative_decode(memory[:, 1:], target=target[:, 1:])

        if self.refine_iters:
            # For iterative refinement, we always use a 'cloze' mask.
            # We can derive it from the AR forward mask by unmasking the token context to the right.
            query_mask = torch.triu(torch.full((num_steps, num_steps + 1), float('-inf'), device=self._device), 1)
            query_mask[torch.triu(torch.ones(num_steps, num_steps + 1, dtype=torch.bool, device=self._device), 2)] = 0
            query_mask_extend[
                torch.triu(torch.ones(num_steps, num_steps + 1, dtype=torch.bool, device=self._device), 2)] = float(
                '-inf')
            bos = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=self._device)
            for i in range(self.refine_iters):
                # Prior context is the previous output.
                tgt_in = torch.cat([bos, logits.argmax(-1)], dim=1)
                # if target is not None:  # if use the gt
                #     tgt_in = target
                tgt_padding_mask = torch.where((tgt_in == self.eos_id).cumsum(-1) > 0, 1.0, 0.0)
                # mask tokens beyond the first EOS token.
                tgt_padding_mask = torch.cat([torch.full((bs, 1), 0.0, device=self._device), tgt_padding_mask[:, :-1]],
                                             dim=1)
                if target is not None:
                    tgt_padding_mask = padding_mask[:, :tgt_in.shape[1]]
                tgt_out = self.decode(tgt_in, memory[:, 1:] if self.len_token else memory, tgt_mask,
                                      tgt_padding_mask=torch.cat([tgt_padding_mask, tgt_padding_mask], dim=-1),
                                      tgt_query=tgt_query,
                                      tgt_query_mask=torch.cat([query_mask[:, :tgt_in.shape[1]],
                                                                query_mask_extend[:, :tgt_in.shape[1]]], dim=1),
                                      max_length=tgt_in.shape[1] - 1,
                                      tgt_extend=tgt_extend)
                logits = self.head(tgt_out)

        return padding_pred, logits

    def forward_logits_loss(self, images: Tensor, labels: List[str]) -> Tuple[Tensor, Tensor, int]:
        targets = self.tokenizer.encode(labels, self.device, self.max_label_length)
        tgt_out = targets[:, 1:]  # Discard <bos>
        bs, tgt_len = tgt_out.shape
        # exclude <eos> from count
        padding_pred, logits = self.forward(images)
        # padding_pred, logits = self.forward(images, target=targets)     # test with ground truth
        # loss = F.cross_entropy(logits.flatten(end_dim=1), tgt_out.flatten(), ignore_index=self.pad_id)
        loss = F.cross_entropy(logits.flatten(end_dim=1), tgt_out[:, :logits.shape[1]].flatten())
        loss_numel = (tgt_out != self.pad_id).sum()
        # predict the length of labels
        padding_gt = torch.full(tgt_out.shape, 0, dtype=torch.long, device=self._device)
        padding_gt = torch.where(tgt_out == self.pad_id, 1, padding_gt)
        padding_loss = F.cross_entropy(padding_pred.flatten(end_dim=1), padding_gt.flatten())
        total_loss = (1.0 - self.len_ratio) * loss + self.len_ratio * padding_loss

        return padding_pred, logits, total_loss, loss_numel

    def gen_tgt_perms(self, tgt):
        """
        Generate shared permutations for the whole batch.
        This works because the same attention mask can be used for
        the shorter sequences because of the padding mask
        tgt = [BOS],token,token,token,[EOS],[PAD]
        """
        # We don't permute the position of BOS, we permute EOS separately
        max_num_chars = tgt.shape[1] - 2  # minus the [BOS] AND [EOS]
        # Special handling for 1-character sequences
        if max_num_chars == 1:  # [1, 2, 3]
            return torch.arange(3, device=self._device).unsqueeze(0)
        perms = [torch.arange(max_num_chars, device=self._device)] if self.perm_forward else []
        # Additional permutations if needed
        max_perms = math.factorial(max_num_chars)  # the number of permutation
        if self.perm_mirrored:
            max_perms //= 2
        num_gen_perms = min(self.max_gen_perms, max_perms)
        # For 4-char sequence and shorter, generate all permutations
        if max_num_chars < 5:
            # Pool of permutations to sample from.
            if max_num_chars == 4 and self.perm_mirrored:
                selector = [0, 3, 4, 6, 9, 10, 12, 16, 17, 18, 19, 21]  # ?????
            else:
                selector = list(range(max_perms))  # [0, 1, 2, ..., max_perms]
            perm_pool = torch.as_tensor(list(permutations(range(max_num_chars))), device=self._device)[selector]
            if self.perm_forward:
                perm_pool = perm_pool[1:]
            perms = torch.stack(perms)
            if len(perm_pool):
                i = self.rng.choice(len(perm_pool), size=num_gen_perms - len(perms), replace=False)
                perms = torch.cat([perms, perm_pool[i]])
        else:
            perms.extend(
                [torch.randperm(max_num_chars, device=self._device) for _ in range(num_gen_perms - len(perms))])
            perms = torch.stack(perms)
        if self.perm_mirrored:
            #   Add complementary pairs
            comp = perms.flip(-1)
            # Stack in such as way that the pairs are next to each other
            perms = torch.stack([perms, comp]).transpose(0, 1).reshape(-1, max_num_chars)
        # Add position indices of [BOS] and [EOS]
        mask_idx = perms.new_zeros((len(perms), 1))
        eos_idx = perms.new_full((len(perms), 1), max_num_chars + 1)
        perms = torch.cat([mask_idx, perms + 1, eos_idx], dim=1)
        if len(perms) > 1:  # perms[1] = [0, 4, 3, 2, 1, 5] (5-0,1,2,3,4) --> [0, 5, 4, 3, 2, 1]
            perms[1, 1:] = max_num_chars + 1 - torch.arange(max_num_chars + 1, device=self._device)
        return perms

    def generate_attn_masks(self, perm):
        """
        Generate the attention masks given a sequence permutation (includes pos. for bos and eos tokens)
        :param perm: the permutation sequence. i = 0 is always the [BOS]
        :return: attention masks
        """
        sz = perm.shape[0]
        mask = torch.zeros((sz, sz), device=self._device)  # (T+2)x(T+2), include the [BOS] and [EOS]
        extend_mask = torch.zeros((sz, sz), device=self._device)
        """
        E.g. perm = [0, 1, 3, 5, 2, 4, 6]
        i = [0,1,2,3,4,5,6], perm[i] = [0,1,3,5,2,4,6]   0 1 3 2
        masked_keys = [1,3,5,2,4,6]  [0]             0            inf, 0, 0, 0, 0, 0
                      [3,5,2,4,6]    [0,1]            1         
                      [5,2,4,6]      [0,1,3]          3         
                      [2,4,6]        [0,1,3,5]        5                   
        """
        for i in range(sz):
            query_idx = perm[i]

            extend_masked_keys = perm[:i + 1]
            extend_mask[query_idx, extend_masked_keys] = float('-inf')

            masked_keys = perm[i + 1:]
            mask[query_idx, masked_keys] = float('-inf')
        """
        after that, mask shape: (T+2)x(T+2), extend_mask shape: (T+2)x(T+2)
        """
        content_mask = mask[:-1, :].clone()  # (T+1)x(T+2)
        content_extend_mask = extend_mask[:-1, :].clone()  # (T+1)x(T+2)
        content_mask = torch.cat([content_mask, content_extend_mask], dim=1)

        mask[torch.eye(sz, dtype=torch.bool, device=self._device)] = float('-inf')  # mask "self"
        extend_mask[torch.eye(sz, dtype=torch.bool, device=self._device)] = 0.0  # unmask "self"
        query_mask = mask[1:, :]
        query_extend_mask = extend_mask[1:, :]
        query_mask = torch.cat([query_mask, query_extend_mask], dim=1)
        return content_mask, query_mask

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, labels = batch
        bs = images.shape[0]
        tgt = self.tokenizer.encode(labels, self._device, self.max_label_length)
        # tgt = ['BOS', token, token, 'EOS', 'PAD', 'PAD'] bs * (max_length+2)

        # Encode the source sequence
        feature = self.encode(images)
        # Prepare the target sequences (input and output)
        tgt_perms = self.gen_tgt_perms(tgt)
        # Add the [MASK] token for the masked-permutation model
        tgt_extend = torch.full((bs, tgt.shape[1]), self.mask_id, dtype=torch.long, device=self._device)
        # change the token to [PAD] for [PAD] token in tgt; ['MASK','MASK','MASK','PAD','PAD','PAD']
        tgt_extend = torch.where(tgt == self.pad_id, self.pad_id, tgt_extend)
        """
        now, the final tgt_in is (tgt_len = 2 x (max_length + 2))
        ['BOS',TOKEN,TOKEN,'EOS','PAD','PAD','MASK','MASK','MASK','MASK','MASK','MASK','MASK']
        For the soft key_padding_mask, the scale of padding_mask is predicted by the [LEN] token,
        which is in range [0,1] and 1 means masking the corresponding token.
        """
        tgt_in = torch.cat([tgt, tgt_extend], dim=1)
        tgt_out = tgt[:, 1:]  # not include the [BOS] token
        # The [EOS] token is not depended upon by any other token in any permutation ordering
        tgt_padding_mask = torch.full(tgt.shape, 0.0, dtype=torch.float, device=self._device)
        tgt_padding_mask = torch.where(tgt == self.pad_id, 1.0, tgt_padding_mask)
        loss = 0
        loss_numel = 0
        loss_padding = 0
        n = (tgt_out != self.pad_id).sum().item()
        padding_gt = torch.full(tgt_out.shape, 0, dtype=torch.long, device=self._device)
        padding_gt = torch.where(tgt_out == self.pad_id, 1, padding_gt)

        pad_query = self.pos_queries[:, :self.max_label_length + 1].expand(bs, -1, -1)

        if self.use_gt:
            for i, perm in enumerate(tgt_perms):
                if self.len_token:
                    # predict the padding use the length token
                    # padding_mask = padding_pred = self.mlp_head(feature[:, 0])
                    # print(padding_pred)
                    # fuse the len_token with the query
                    pad_query = pad_query + feature[:, 0].unsqueeze(1).expand(-1, self.max_label_length + 1, -1)
                if self.detach:
                    padding_pred = self.pad_predict(pad_query,
                                                    feature[:, 1:].detach() if self.len_token else feature.detach())
                else:
                    padding_pred = self.pad_predict(pad_query, feature[:, 1:] if self.len_token else feature)
                padding_pred = padding_pred.flatten(end_dim=1)
                loss_padding += n * F.cross_entropy(padding_pred, padding_gt.flatten())

                tgt_mask, query_mask = self.generate_attn_masks(perm)
                out = self.decode(tgt_in, feature[:, 1:] if self.len_token else feature, tgt_mask,
                                  torch.concat([tgt_padding_mask, tgt_padding_mask], dim=1),
                                  tgt_query_mask=query_mask)
                logits = self.head(out).flatten(end_dim=1)
                # truncate at the first eos token of the prediction
                loss += n * F.cross_entropy(logits, tgt_out.flatten(), ignore_index=self.pad_id)
                loss_numel += n
                # After the second iteration (i.e. done with canonical and reverse orderings),
                # remove the [EOS] tokens for the succeeding perms
                '''
                if i == 1:
                    tgt_out = torch.where(tgt_out == self.eos_id, self.pad_id, tgt_out)
                    n = (tgt_out != self.pad_id).sum().item()
                '''
        else:
            # use the predicted key padding mask to train
            for i, perm in enumerate(tgt_perms):
                if self.len_token:
                    # generate the padding_mask using the length token
                    padding_mask = padding_pred = self.mlp_head(feature[:, 0])
                    # print(padding_pred)
                padding_pred = self.pad_predict(pad_query, feature[:, 1:] if self.len_token else feature)
                softmax_padding_pred = F.softmax(padding_pred, dim=2)
                padding_mask = torch.cat([torch.full((bs, 1), 0.0, device=self._device), softmax_padding_pred[:, :, 1]],
                                         dim=1)
                padding_pred = padding_pred.flatten(end_dim=1)
                loss_padding += n * F.cross_entropy(padding_pred, padding_gt.flatten())

                tgt_mask, query_mask = self.generate_attn_masks(perm)
                # use the predicted mask for padding [MASK]
                out = self.decode(tgt_in, feature[:, 1:] if self.len_token else feature, tgt_mask,
                                  torch.concat([tgt_padding_mask, padding_mask], dim=1),
                                  tgt_query_mask=query_mask)
                # use the predicted mask both for gt tokens and padding [MASK]
                '''
                out = self.decode(tgt_in, feature[:, 1:] if self.len_token else feature, tgt_mask,
                                  torch.concat([padding_mask, padding_mask], dim=1),
                                  tgt_query_mask=query_mask)
                '''
                logits = self.head(out).flatten(end_dim=1)
                loss += n * F.cross_entropy(logits, tgt_out.flatten(), ignore_index=self.pad_id)
                # loss += n * F.cross_entropy(logits, tgt_out.flatten())
                loss_numel += n
                # After the second iteration (i.e. done with canonical and reverse orderings),
                # remove the [EOS] tokens for the succeeding perms
                if i == 1:
                    tgt_out = torch.where(tgt_out == self.eos_id, self.pad_id, tgt_out)
                    n = (tgt_out != self.pad_id).sum().item()

        loss /= loss_numel
        loss_padding /= loss_numel
        print('loss_padding', loss_padding)
        loss = loss + self.len_ratio * loss_padding
        # loss = (1 - self.len_ratio) * loss + self.len_ratio * loss_padding

        self.log('loss', loss)
        return loss
