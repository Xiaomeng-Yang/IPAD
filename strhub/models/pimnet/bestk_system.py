from functools import partial
from typing import Sequence, Any, Optional
import torch
import math
from typing import Optional, Tuple, List
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dataclasses import dataclass
from nltk import edit_distance
from pytorch_lightning.utilities.types import STEP_OUTPUT
from timm.models.helpers import named_apply

from strhub.data.utils import PIMNetTokenizer
from strhub.models.base import BaseSystem
from strhub.models.utils import init_weights
from .modules import Encoder, TokenEmbedding
from .at_decoder import AT_Decoder
from .parallel_decoder import Decoder


@dataclass
class BatchResult:
    num_samples: int
    correct: int
    ned: float
    confidence: float
    label_length: int
    pred_list: List
    gt_list: List


class PIMNet(BaseSystem):
    def __init__(self, charset_train: str, charset_test: str, max_label_length: int,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float,
                 img_size: Sequence[int], patch_size: Sequence[int], embed_dim: int,
                 enc_num_heads: int, enc_mlp_ratio: int, enc_depth: int, dec_num_heads: int,
                 num_iter: int, dropout: float,
                 **kwargs: Any) -> None:
        tokenizer = PIMNetTokenizer(charset_train)
        super().__init__(tokenizer, charset_test, batch_size, lr, warmup_pct, weight_decay)
        self.eos_id = tokenizer.eos_id
        self.mask_id = tokenizer.mask_id
        self.pad_id = tokenizer.pad_id

        self.save_hyperparameters()
        self.num_iter = num_iter
        self.max_len = max_label_length
        self.top_k = math.ceil(self.max_len / self.num_iter)
        self.keep_num = 2
        self.dec_num_heads = dec_num_heads

        self.text_embed = TokenEmbedding(len(self.tokenizer), embed_dim)
        self.pos_embed = nn.Parameter(torch.Tensor(1, self.max_len, embed_dim))
        # We don't predict <pad>
        # self.head = nn.Linear(embed_dim, len(self.tokenizer) - 1)
        self.head = nn.Linear(embed_dim, len(self.tokenizer) - 2)   # don't predict <pad> or <mask>
        self.dropout = nn.Dropout(p=dropout)
        self.encoder = Encoder(img_size, patch_size, embed_dim=embed_dim, depth=enc_depth,
                               num_heads=enc_num_heads, mlp_ratio=enc_mlp_ratio)

        self.at_decoder = AT_Decoder(d_model=embed_dim, nhead=dec_num_heads)
        self.decoder = Decoder(d_model=embed_dim, nhead=dec_num_heads)
        # Encoder has its own init
        named_apply(partial(init_weights, exclude=['encoder']), self)
        nn.init.trunc_normal_(self.pos_embed, std=.02)

    def encode(self, img: torch.Tensor):
        return self.encoder(img)

    def decode(self, tgt: torch.Tensor, feature_map: torch.Tensor, tgt_mask: Optional[Tensor] = None,
               tgt_padding_mask: Optional[Tensor] = None):
        """
        Autoregressive decoder
        """
        N, L = tgt.shape
        null_ctx = self.text_embed(tgt[:, :1])
        tgt_emb = self.pos_embed[:, :L-1] + self.text_embed(tgt[:, 1:])
        tgt_emb = self.dropout(torch.cat([null_ctx, tgt_emb], dim=1))
        return self.at_decoder(tgt_emb, feature_map, tgt_mask, tgt_padding_mask)

    def inference_beam_search(self, feature_map: torch.Tensor):
        """
        Easy first decoding strategy
        :param feature_map:
        :param input_labels
        """
        bs = feature_map.shape[0]   # the batch size
        # the first iteration, generates the predictions based on masks
        tgt_tokens = torch.full((bs, self.max_len), self.mask_id, dtype=torch.long, device=self._device)
        pred_tgt_tokens = torch.full((bs, self.max_len), self.mask_id, dtype=torch.long, device=self._device)
        keep_token_probs = torch.zeros([bs, self.keep_num**self.top_k], dtype=torch.float, device=self._device)
        keep_tokens = torch.full((bs, self.max_len, self.keep_num**self.top_k),
                                 self.mask_id, dtype=torch.long, device=self._device)

        predicts_embed = self.text_embed(tgt_tokens) + self.pos_embed
        outputs, alphas = self.decoder(predicts_embed, feature_map, key_padding_mask=None, attn_mask=None)
        new_token_logits = self.head(outputs).float()
        new_tgt_tokens = torch.argmax(new_token_logits, dim=-1)
        token_probs = new_token_logits.softmax(-1)  # N*T*num_token

        # keep keep_num possible tokens for each position
        new_keep_tokens = torch.topk(new_token_logits, k=self.keep_num, dim=-1).indices
        token_probs = torch.topk(token_probs, k=self.keep_num, dim=-1).values

        max_token_probs = torch.max(token_probs, dim=-1).values  # N*T
        top_values, top_indices = max_token_probs.topk(self.top_k, dim=1)  # get the top-k best position
        update_idx = torch.full((bs, self.max_len), False, dtype=torch.bool, device=self._device)
        update_idx[torch.arange(bs).unsqueeze(1), top_indices] = True

        # we can get the keep_tokens for following iteration
        cur_keep_tokens = new_keep_tokens[update_idx].reshape(bs, self.top_k, self.keep_num)
        cur_token_probs = token_probs[update_idx].reshape(bs, self.top_k, self.keep_num)
        for img_idx in range(bs):
            temp = torch.cartesian_prod(*cur_keep_tokens[img_idx])
            temp_prob = torch.sum(torch.cartesian_prod(*cur_token_probs[img_idx]), dim=-1)
            keep_tokens[img_idx, update_idx[img_idx]] = torch.transpose(temp, 0, 1)
            keep_token_probs[img_idx] = temp_prob

        # if there is only 1 iteration:
        pred_tgt_tokens = torch.where(update_idx, new_tgt_tokens, pred_tgt_tokens)

        for i in range(1, self.num_iter):
            best_sum = torch.zeros(bs, dtype=torch.float, device=self._device)
            temp_token_probs = torch.zeros([bs, self.keep_num ** self.top_k], dtype=torch.float, device=self._device)
            temp_keep_tokens = torch.full((bs, self.max_len, self.keep_num ** self.top_k),
                                          self.mask_id, dtype=torch.long, device=self._device)
            for prob_idx in range(self.keep_num ** self.top_k):
                tgt_tokens = keep_tokens[:, :, prob_idx]
                # generate the corresponding mask and embed
                key_padding_mask = ((tgt_tokens == self.eos_id).cumsum(-1) > 0)
                mask_mask = (tgt_tokens == self.mask_id)
                key_padding_mask = key_padding_mask | mask_mask

                # checking for all true case
                for row in range(bs):
                    if key_padding_mask[row, :].all():
                        key_padding_mask[row, :] = False

                predicts_embed = self.text_embed(tgt_tokens) + self.pos_embed
                outputs, alphas = self.decoder(predicts_embed, feature_map, key_padding_mask, attn_mask=None)
                new_token_logits = self.head(outputs).float()
                new_tgt_tokens = torch.argmax(new_token_logits, dim=-1)
                token_probs = new_token_logits.softmax(-1)  # N*T*num_token

                # first calculate the current_probs
                max_token_probs = torch.max(token_probs, dim=-1).values  # N*T
                max_token_probs = torch.where(tgt_tokens == self.mask_id, max_token_probs,
                                              torch.zeros_like(max_token_probs))
                top_values, top_indices = max_token_probs.topk(self.top_k, dim=1)  # get the top-k best position
                cur_sum = torch.sum(top_values, dim=-1) + keep_token_probs[:, prob_idx]
                update_img = torch.where(cur_sum > best_sum)[0]

                # update the values
                best_sum = torch.where(cur_sum > best_sum, cur_sum, best_sum)
                temp_keep_tokens[update_img] = torch.tile(tgt_tokens[update_img].unsqueeze(dim=2),
                                                          [1, 1, self.keep_num**self.top_k])
                temp_token_probs[update_img] = keep_token_probs[update_img]
                pred_tgt_tokens[update_img] = tgt_tokens[update_img]
                update_idx = torch.full((bs, self.max_len), False, dtype=torch.bool, device=self._device)
                update_idx[torch.arange(bs).unsqueeze(1), top_indices] = True

                pred_tgt_tokens[update_img] = torch.where(update_idx[update_img],
                                                          new_tgt_tokens[update_img],
                                                          pred_tgt_tokens[update_img])

                # keep keep_num possible tokens for each position
                new_keep_tokens = torch.topk(new_token_logits, k=self.keep_num, dim=-1).indices
                token_probs = torch.topk(token_probs, k=self.keep_num, dim=-1).values

                # we can get the keep_tokens for following iteration
                cur_keep_tokens = new_keep_tokens[update_idx].reshape(bs, self.top_k, self.keep_num)
                cur_token_probs = token_probs[update_idx].reshape(bs, self.top_k, self.keep_num)
                for img_idx in update_img:
                    temp = torch.cartesian_prod(*cur_keep_tokens[img_idx])
                    temp_prob = torch.sum(torch.cartesian_prod(*cur_token_probs[img_idx]), dim=-1)
                    temp_keep_tokens[img_idx, update_idx[img_idx]] = torch.transpose(temp, 0, 1)
                    temp_token_probs[img_idx] += temp_prob

            # finish the processing of all possible combinations, pred_tgt_tokens now contain the best candidates
            keep_token_probs = temp_token_probs
            keep_tokens = temp_keep_tokens

        return pred_tgt_tokens

    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        # encode the images
        feature_map = self.encode(images)
        # iterative_decoder
        pred_tokens = self.inference_beam_search(feature_map)
        return pred_tokens

    def _eval_step(self, batch, validation: bool) -> Optional[STEP_OUTPUT]:
        images, labels = batch
        confidence = 0.0
        correct = 0
        total = 0
        ned = 0
        label_length = 0
        pred_tokens = self.forward(images)
        preds = self.tokenizer.bestk_decode(pred_tokens)
        pred_list, gt_list = [], []
        i = 0
        for pred, gt in zip(preds, labels):
            pred = self.charset_adapter(pred)
            ned += edit_distance(pred, gt) / max(len(pred), len(gt))
            pred_list.append(pred)
            gt_list.append(gt)
            if pred == gt:
                correct += 1
            total += 1
            label_length += len(pred)
            i += 1
        return dict(output=BatchResult(total, correct, ned, confidence, label_length, pred_list, gt_list))

    def forward_logits_loss(self, images: Tensor, labels: List[str]) -> Tuple[Tensor, Tensor, int]:
        bs = images.shape[0]
        targets = self.tokenizer.encode(labels, self._device, self.max_len)

        tgt_in = torch.full((bs, self.max_len), self.mask_id, dtype=torch.long, device=self._device)
        # encode the images
        feature_map = self.encode(images)
        # Decode using the parallel and autoregressive decoder
        iter_logits, _, _, nat_glimpses = self.iterative_decode(feature_map)

        at_glimpses = torch.zeros_like(nat_glimpses, device=self._device)
        for i in range(self.max_len):
            j = i + 1
            at_glimpses = self.decode(tgt_in[:, :j], feature_map)
            # the next token probability is in the output's ith token position
            at_logits = self.head(at_glimpses)
            # greedy decode. add the next token index to the target input
            if j < self.max_len:
                tgt_in[:, j] = at_logits[:, j-1, :].argmax(-1)

        at_loss = F.cross_entropy(at_logits.flatten(end_dim=1), targets.flatten(), ignore_index=self.pad_id)
        iter_loss = F.cross_entropy(iter_logits.flatten(end_dim=1), targets.flatten(), ignore_index=self.pad_id)
        mimic_loss = F.cosine_embedding_loss(at_glimpses.detach().flatten(start_dim=1),
                                             nat_glimpses.flatten(start_dim=1),
                                             torch.ones(bs, device=self._device), reduction='mean')
        loss = at_loss + iter_loss + mimic_loss
        loss_numel = (targets != self.pad_id).sum()
        return iter_logits, loss, loss_numel

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, labels = batch
        bs = images.shape[0]
        # generate the label's tokens
        tgt = self.tokenizer.encode(labels, self._device, self.max_len)
        # tgt = [l,a,b,l,e,[E],[P],[P]] bs * max_len
        # Encode the source sequence
        feature_map = self.encode(images)
        # Prepare the target sequences (input and output)
        tgt_in = tgt[:, :-1]    # bs * max_len-1
        tgt_in = torch.cat([torch.full((bs, 1), self.mask_id, dtype=torch.long, device=self._device), tgt_in], dim=1)

        # The [EOS] token is not depended upon by any other token in any permutation ordering
        tgt_padding_mask = (tgt_in == self.pad_id) | (tgt_in == self.eos_id)
        tgt_mask = torch.triu(torch.full((self.max_len, self.max_len), float('-inf'), device=self._device), 1)

        # Decode using the parallel and autoregressive decoder
        at_glimpses = self.decode(tgt_in, feature_map, tgt_mask=tgt_mask, tgt_padding_mask=tgt_padding_mask)
        at_logits = self.head(at_glimpses).flatten(end_dim=1)

        iter_logits, _, _, nat_glimpses = self.iterative_decode(feature_map, input_labels=tgt)

        # n = (tgt != self.pad_id).sum().item()
        at_loss = F.cross_entropy(at_logits, tgt.flatten(), ignore_index=self.pad_id)
        iter_loss = F.cross_entropy(iter_logits.flatten(end_dim=1), tgt.flatten(), ignore_index=self.pad_id)
        mimic_loss = F.cosine_embedding_loss(at_glimpses.detach().flatten(start_dim=1),
                                             nat_glimpses.flatten(start_dim=1),
                                             torch.ones(bs, device=self._device), reduction='mean')
        loss = at_loss + iter_loss + mimic_loss

        self.log('loss', loss)
        return loss
