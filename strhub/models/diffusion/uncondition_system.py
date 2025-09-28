import math
from .utils import *
from torch import nn
from torch import Tensor
from functools import partial
from typing import Sequence, Any, Optional, Tuple, List
from timm.models.helpers import named_apply
from pytorch_lightning.utilities.types import STEP_OUTPUT

from strhub.data.utils import PIMNetTokenizer
from strhub.models.base import BaseSystem
from strhub.models.utils import init_weights

from .modules import TokenEmbedding, Encoder, Decoder, DecoderLayerEdit


class DDPIMNet(BaseSystem):
    def __init__(self, charset_train: str, charset_test: str, max_label_length: int,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float,
                 img_size: Sequence[int], patch_size: Sequence[int], embed_dim: int,
                 enc_num_heads: int, enc_mlp_ratio: int, enc_depth: int, dec_num_heads: int,
                 dec_depth: int, dec_mlp_ratio: int, time_step: int, dropout: float,
                 len_token: bool, len_ratio: float, num_iter: int, **kwargs: Any) -> None:
        tokenizer = PIMNetTokenizer(charset_train)
        super().__init__(tokenizer, charset_test, batch_size, lr, warmup_pct, weight_decay)
        self.eos_id = tokenizer.eos_id
        self.mask_id = tokenizer.mask_id
        self.pad_id = tokenizer.pad_id
        self.save_hyperparameters()

        self.time_step = time_step
        self.num_token = len(self.tokenizer)
        self.len_token = len_token
        self.len_ratio = len_ratio
        self.max_len = max_label_length
        self.num_iter = num_iter
        self.top_k = math.ceil(self.max_len / self.num_iter)
        self.dec_num_heads = dec_num_heads

        # self._init_linear_schedule()
        self._init_mask_schedule()

        self.text_embed = TokenEmbedding(self.num_token, embed_dim)
        self.pos_embed = nn.Parameter(torch.Tensor(1, self.max_len, embed_dim))
        self.head = nn.Linear(embed_dim, self.num_token - 2)    # do not predict the <pad> and <mask> token
        # encoder provides the image features
        if self.len_token:
            self.encoder = Encoder(img_size, patch_size, embed_dim=embed_dim, depth=enc_depth,
                                   num_heads=enc_num_heads, mlp_ratio=enc_mlp_ratio, class_token=True)
            self.length_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, self.max_len)
            )
        else:
            self.encoder = Encoder(img_size, patch_size, embed_dim=embed_dim, depth=enc_depth,
                                   num_heads=enc_num_heads, mlp_ratio=enc_mlp_ratio, class_token=False)
        self.scaling_factors = nn.Parameter(torch.as_tensor(self.get_default_scaling_factors(),
                                                            dtype=torch.float32))
        # the diffusion training decoder
        decoder_layer = DecoderLayerEdit(embed_dim, dec_num_heads, time_step, embed_dim * dec_mlp_ratio, dropout)
        self.decoder = Decoder(decoder_layer, num_layers=dec_depth, norm=nn.LayerNorm(embed_dim))
        self.scaling_factors = nn.Parameter(torch.as_tensor(self.get_default_scaling_factors(),
                                                            dtype=torch.float32))
        # initialize
        named_apply(partial(init_weights, exclude=['encoder']), self)
        nn.init.trunc_normal_(self.pos_embed, std=.02)

    def get_default_scaling_factors(self):
        return [-0.1]*(self.time_step)

    def _init_linear_schedule(self):
        # store the transfer probabilities
        at, bt, ct, att, btt, ctt = linear_schedule(self.time_step, N=self.num_token - 2)  # alpha schedule
        self.register_buffer('log_at', torch.log(torch.tensor(at)).float())
        self.register_buffer('log_bt', torch.log(torch.tensor(bt)).float())
        self.register_buffer('log_ct', torch.log(torch.tensor(ct)).float())
        self.register_buffer('log_att', torch.log(torch.tensor(att)).float())
        self.register_buffer('log_btt', torch.log(torch.tensor(btt)).float())
        self.register_buffer('log_ctt', torch.log(torch.tensor(ctt)).float())
        self.register_buffer('log_1_min_ct', torch.log(1 - self.log_ct.exp() + 1e-40))
        self.register_buffer('log_1_min_ctt', torch.log(1 - self.log_ctt.exp() + 1e-40))

    def _init_mask_schedule(self):
        at, ct, att, ctt = mask_schedule(self.time_step)
        self.register_buffer('log_at', torch.log(torch.tensor(at)).float())
        self.register_buffer('log_att', torch.log(torch.tensor(att)).float())
        self.register_buffer('log_ct', torch.log(torch.tensor(ct)).float())
        self.register_buffer('log_ctt', torch.log(torch.tensor(ctt)).float())
        self.register_buffer('log_1_min_ct', torch.log(1 - self.log_ct.exp() + 1e-40))
        self.register_buffer('log_1_min_ctt', torch.log(1 - self.log_ctt.exp() + 1e-40))

    def q_pred(self, log_x_start, t):  # q(xt|x0)
        # log_x_start can be onehot or not
        t = (t + (self.time_step + 1)) % (self.time_step + 1)
        log_att = extract(self.log_att, t, log_x_start.shape)  # at~
        log_btt = extract(self.log_btt, t, log_x_start.shape)  # bt~
        log_ctt = extract(self.log_ctt, t, log_x_start.shape)  # ct~
        log_1_min_ctt = extract(self.log_1_min_ctt, t, log_x_start.shape)  # 1-ct~

        # log_probs = torch.zeros(log_x_start.size()).type_as(log_x_start)
        p1 = log_add_exp(log_x_start[:, :-1, :] + log_att, log_btt)
        p2 = log_add_exp(log_x_start[:, -1:, :] + log_1_min_ctt, log_ctt)
        return torch.cat([p1, p2], dim=1)

    def q_mask_pred(self, log_x_start, t):
        t = (t + (self.time_step + 1)) % (self.time_step + 1)
        log_att = extract(self.log_att, t, log_x_start.shape)  # at~
        log_ctt = extract(self.log_ctt, t, log_x_start.shape)  # ct~
        log_1_min_ctt = extract(self.log_1_min_ctt, t, log_x_start.shape)  # 1-at~
        p1 = log_x_start[:, :1, :] + log_att
        p2 = log_x_start[:, 1:-1, :]
        p3 = log_add_exp(log_x_start[:, -1:, :] + log_1_min_ctt, log_ctt)
        return torch.cat([p1, p2, p3], dim=1)

    def log_sample_categorical(self, logits):  # use gumbel to sample onehot vector from log probability
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = idx_to_log_onehot(sample, self.num_token - 1)
        return log_sample

    def encode(self, img: torch.Tensor):
        return self.encoder(img)

    def decode(self, t: torch.Tensor, tgt: torch.Tensor, feature_map: torch.Tensor,
               key_padding_mask: Optional[Tensor] = None):
        if key_padding_mask is None:
            key_padding_mask = ((tgt == self.eos_id).cumsum(-1) > 0)

        mask_mask = (tgt == self.mask_id)
        key_padding_mask = key_padding_mask | mask_mask
        # checking for all true case
        for row in range(tgt.shape[0]):
            if key_padding_mask[row, :].all():
                key_padding_mask[row, :] = False

        tgt = self.text_embed(tgt) + self.pos_embed
        return self.decoder(t, tgt, feature_map, key_padding_mask, attn_mask=None)

    def decode_uncondition(self, t: torch.Tensor, tgt: torch.Tensor, feature_map: torch.Tensor,
               key_padding_mask: Optional[Tensor] = None):
        if key_padding_mask is None:
            key_padding_mask = ((tgt == self.eos_id).cumsum(-1) > 0)

        mask_mask = (tgt == self.mask_id)
        key_padding_mask = key_padding_mask | mask_mask
        # checking for all true case
        for row in range(tgt.shape[0]):
            if key_padding_mask[row, :].all():
                key_padding_mask[row, :] = False

        tgt = self.text_embed(tgt) + self.pos_embed
        feature_map = self.text_embed(feature_map)
        return self.decoder(t, tgt, feature_map, key_padding_mask, attn_mask=None, uncondition=True)

    def iterative_decode(self, feature_map: torch.Tensor, input_labels: Optional[Tensor] = None,
                         pred_len: Optional[Tensor] = None):
        """
        Easy first decoding strategy
        :param feature_map:
        :param input_labels
        :param pred_len
        """
        bs = feature_map.shape[0]   # the batch size
        C = feature_map.shape[-1]

        tgt_tokens = torch.full((bs, self.max_len), self.mask_id, dtype=torch.long, device=self._device)
        pred_tgt_tokens = torch.full((bs, self.max_len), self.mask_id, dtype=torch.long, device=self._device)
        token_logits = torch.zeros([bs, self.max_len, len(self.tokenizer) - 2],
                                   dtype=torch.float, device=self._device)
        if pred_len is not None:
            tgt_tokens.scatter_(dim=1, index=pred_len, value=self.eos_id)

        if self.num_iter > self.time_step:
            self.num_iter = self.time_step

        for i in range(self.num_iter):
            t = self.time_step - 1 - i*(self.time_step // self.num_iter)
            t = torch.full((bs,), t, dtype=torch.long, device=self._device)
            outputs = self.decode(t, tgt_tokens, feature_map)
            out_uncondition = self.decode_uncondition(t, tgt_tokens, tgt_tokens)
            # combine the feature-level
            
            new_token_logits = self.head(outputs).float()
            new_token_uncondition = self.head(out_uncondition).float()
            new_token_logits = (1 + self.scaling_factors[i]) * new_token_logits\
                               - self.scaling_factors[i] * new_token_uncondition
                               
            token_probs = new_token_logits.softmax(-1)

            new_tgt_tokens = torch.argmax(new_token_logits, dim=-1)
            # only predict the mask ones
            token_probs = torch.max(token_probs, dim=-1).values     # N*T
            token_probs = torch.where(tgt_tokens == self.mask_id, token_probs, torch.zeros_like(token_probs))

            top_tuple = token_probs.topk(self.top_k, dim=1)     # get the top-k best position
            kth = torch.min(top_tuple.values, dim=1, keepdim=True).values
            update_idx = torch.greater_equal(token_probs, kth)

            logits_update_idx = torch.tile(update_idx.unsqueeze(dim=2), [1, 1, len(self.tokenizer) - 2])

            if input_labels is not None:    # is training
                tgt_tokens = torch.where(update_idx, input_labels, tgt_tokens)
                pred_tgt_tokens = torch.where(update_idx, new_tgt_tokens, pred_tgt_tokens)

            else:   # testing
                tgt_tokens = torch.where(update_idx, new_tgt_tokens, tgt_tokens)
                pred_tgt_tokens = tgt_tokens

            token_logits = torch.where(logits_update_idx, new_token_logits, token_logits)

        return token_logits, pred_tgt_tokens

    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        # encode the source images
        print(self.scaling_factors)
        bs = images.shape[0]
        feature_map = self.encode(images)
        if self.len_token:
            len_logits = self.length_head(feature_map[:, 0])
            pred_len = len_logits.argmax(dim=1).view(-1, 1)
            feature_map = feature_map[:, 1:]
            logits, _ = self.iterative_decode(feature_map, pred_len=pred_len)
        else:
            # denoise from the all-masked tokens, use the easy-first decoding procedure
            logits, _ = self.iterative_decode(feature_map)
        return logits

    def forward_logits_loss(self, images: Tensor, labels: List[str]) -> Tuple[Tensor, Tensor, int]:
        # tokenizing to get the target
        bs = images.shape[0]
        tgt = self.tokenizer.encode(labels, self._device, self.max_len)
        loss_numel = (tgt != self.pad_id).sum()
        feature_map = self.encode(images)
        if self.len_token:
            len_gt = torch.full((bs, tgt.shape[1]), 0.0, dtype=torch.float, device=self._device)
            len_gt = torch.where(tgt == self.eos_id, 1.0, len_gt)
            len_logits = self.length_head(feature_map[:, 0])
            pred_len = len_logits.argmax(dim=1).view(-1, 1)
            feature_map = feature_map[:, 1:]
            logits, _ = self.iterative_decode(feature_map, pred_len=pred_len)

            loss_len = F.cross_entropy(len_logits, len_gt)
            loss_denoise = F.cross_entropy(logits.flatten(end_dim=1), tgt.flatten(), ignore_index=self.pad_id)
            loss = self.len_ratio * loss_len + loss_denoise
        else:
            logits, _ = self.iterative_decode(feature_map)
            loss_denoise = F.cross_entropy(logits.flatten(end_dim=1), tgt.flatten(), ignore_index=self.pad_id)
            loss = loss_denoise
        return logits, loss, loss_numel

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, labels = batch
        bs = images.shape[0]
        # tokenizing using the tokenizer
        tgt = self.tokenizer.encode(labels, self._device, self.max_len)
        tgt_mask = self.tokenizer.encode_mask(labels, self._device, self.max_len)
        # encode the source images
        feature_map = self.encode(images)
        if self.len_token:
            len_gt = torch.full((bs, tgt.shape[1]), 0.0, dtype=torch.float, device=self._device)
            len_gt = torch.where(tgt == self.eos_id, 1.0, len_gt)
            # the length is predicted by the [len] token
            len_logits = self.length_head(feature_map[:, 0])  # (bs, self.max_label_length + 1)
            # the length loss
            loss_len = F.cross_entropy(len_logits, len_gt)
            feature_map = feature_map[:, 1:]

        # --------------- the noising process, generate the noised tokens --------------------
        # sample a t in [1, T] uniformly
        t, _ = sample_time(bs, torch.tensor(self.time_step).to(torch.int), self._device)
        # generate the one-hot token for x0
        x0 = idx_to_log_onehot(tgt_mask, self.num_token - 1)
        # get the xt using the q(xt|x0)
        # log_xt = self.log_sample_categorical(self.q_pred(x0, t))
        log_xt = self.log_sample_categorical(self.q_mask_pred(x0, t))
        xt = log_onehot_to_idx(log_xt)
        # print(xt.shape)

        # --------------- the denoising process, predict the x0 using the xt and image feature -------------------
        # only calculate loss for the masked tokens
        key_padding_mask = (tgt == self.pad_id) | (tgt == self.eos_id)
        out = self.decode(t, xt, feature_map, key_padding_mask=key_padding_mask)
        out_uncondition = self.decode_uncondition(t, xt, xt, key_padding_mask=key_padding_mask)
        
        logits_condition = self.head(out) 
        logits_uncondition = self.head(out_uncondition)
        
        logits = (1 + self.scaling_factors[t]).view(-1, 1, 1) * logits_condition.detach() - \
                 self.scaling_factors[t].view(-1, 1, 1) * logits_uncondition.detach()
        loss_uncondition = F.cross_entropy(logits_uncondition.flatten(end_dim=1), tgt.flatten(),
                                           ignore_index=self.pad_id)
        loss_condition = F.cross_entropy(logits_condition.flatten(end_dim=1), tgt.flatten(),
                                         ignore_index=self.pad_id)
        loss_denoise = F.cross_entropy(logits.flatten(end_dim=1), tgt.flatten(), ignore_index=self.pad_id)
        loss = 0.5*loss_denoise + loss_condition + 0.1*loss_uncondition
        
        self.log('loss', loss)
        return loss
