from functools import partial
from typing import Sequence, Any, Optional
from einops import repeat
import torch
import math
from typing import Optional, Tuple, List
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pytorch_lightning.utilities.types import STEP_OUTPUT
from timm.models.helpers import named_apply

from strhub.data.utils import PIMNetTokenizer
from strhub.models.base import BaseSystem
from strhub.models.utils import init_weights
from .modules import Encoder, TokenEmbedding
from .at_decoder import AT_DecoderEdit


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
        self.max_len = max_label_length
        self.dec_num_heads = dec_num_heads

        self.text_embed = TokenEmbedding(len(self.tokenizer), embed_dim)
        self.pos_embed = nn.Parameter(torch.Tensor(1, self.max_len, embed_dim))
        # We don't predict <pad>
        self.head = nn.Linear(embed_dim, len(self.tokenizer) - 2)   # don't predict <pad>
        self.dropout = nn.Dropout(p=dropout)
        self.encoder = Encoder(img_size, patch_size, embed_dim=embed_dim, depth=enc_depth,
                               num_heads=enc_num_heads, mlp_ratio=enc_mlp_ratio)

        self.at_decoder = AT_DecoderEdit(d_model=embed_dim, nhead=dec_num_heads)
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

    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        # encode the images
        feature_map = self.encode(images)
        # at decoder
        tgt_in = torch.full((bs, self.max_len), self.mask_id, dtype=torch.long, device=self._device)
        for i in range(self.max_len):
            j = i + 1
            at_glimpses = self.decode(tgt_in[:, :j], feature_map)
            # the next token probability is in the output's ith token position
            at_logits = self.head(at_glimpses)
            # greedy decode. add the next token index to the target input
            if j < self.max_len:
                tgt_in[:, j] = at_logits[:, j-1, :].argmax(-1)
        return at_logits

    def forward_logits_loss(self, images: Tensor, labels: List[str]) -> Tuple[Tensor, Tensor, int]:
        bs = images.shape[0]
        targets = self.tokenizer.encode(labels, self._device, self.max_len)

        tgt_in = torch.full((bs, self.max_len), self.mask_id, dtype=torch.long, device=self._device)
        # encode the images
        feature_map = self.encode(images)

        for i in range(self.max_len):
            j = i + 1
            at_glimpses = self.decode(tgt_in[:, :j], feature_map)
            # the next token probability is in the output's ith token position
            at_logits = self.head(at_glimpses)
            # greedy decode. add the next token index to the target input
            if j < self.max_len:
                tgt_in[:, j] = at_logits[:, j-1, :].argmax(-1)

        loss = F.cross_entropy(at_logits.flatten(end_dim=1), targets.flatten(), ignore_index=self.pad_id)
        loss_numel = (targets != self.pad_id).sum()
        return at_logits, loss, loss_numel

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

        loss = F.cross_entropy(at_logits, tgt.flatten(), ignore_index=self.pad_id)
        self.log('loss', loss)
        return loss
