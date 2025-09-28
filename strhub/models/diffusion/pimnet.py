import math
import torch
from typing import Optional
from torch import nn
from torch import Tensor
from .modules import Decoder, Encoder, TokenEmbedding


class PIMNet(nn.Module):
    """
    The baseline architecture
    """
    def __init__(self, img_size, patch_size, embed_dim, enc_depth, enc_num_heads,
                 enc_mlp_ratio, dec_num_heads, num_iter, max_len, mask_id, pad_id,
                 num_classes):
        super().__init__()
        self.mask_id = mask_id
        self.pad_id = pad_id
        self.num_classes = num_classes

        self.num_iter = num_iter
        self.max_len = max_len
        self.top_k = math.ceil(max_len / num_iter)
        self.text_embed = TokenEmbedding(num_classes, embed_dim)
        self.pos_embed = nn.Parameter(torch.Tensor(1, max_len, embed_dim))
        self.encoder = Encoder(img_size, patch_size, embed_dim=embed_dim, depth=enc_depth,
                               num_heads=enc_num_heads, mlp_ratio=enc_mlp_ratio)

        self.decoder = Decoder(d_model=embed_dim, nhead=dec_num_heads)
        # self.head = nn.Linear

    def forward(self, images: Tensor, target: Optional[Tensor] = None):
        # encode the images
        feature_map = self.encode(images)
        bs, C = feature_map.shape[0], feature_map.shape[-1]

        tgt_tokens = torch.full((bs, self.max_len), self.mask_id, dtype=torch.long)
        pred_tgt_tokens = torch.full((bs, self.max_len), self.mask_id, dtype=torch.long)
        token_logits = torch.zeros([bs, self.max_len, len(self.tokenizer) - 2], dtype=torch.float)
        final_ffn = torch.zeros([bs, self.max_len, C], dtype=torch.float)

        for i in range(self.num_iter):
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
            token_probs = new_token_logits.softmax(-1)
            new_ffn = outputs

            new_tgt_tokens = torch.argmax(new_token_logits, dim=-1)
            # only predict the mask ones
            token_probs = torch.max(token_probs, dim=-1).values     # N*T
            token_probs = torch.where(tgt_tokens == self.mask_id, token_probs, torch.zeros_like(token_probs))

            top_tuple = token_probs.topk(self.top_k, dim=1)     # get the top-k best position
            kth = torch.min(top_tuple.values, dim=1, keepdim=True).values
            update_idx = torch.greater_equal(token_probs, kth)

            logits_update_idx = torch.tile(update_idx.unsqueeze(dim=2), [1, 1, len(self.tokenizer) - 2])
            ffn_update_idx = torch.tile(update_idx.unsqueeze(dim=2), [1, 1, C])

            if target is not None:    # is training
                tgt_tokens = torch.where(update_idx, target, tgt_tokens)
                pred_tgt_tokens = torch.where(update_idx, new_tgt_tokens, pred_tgt_tokens)

            else:   # testing
                tgt_tokens = torch.where(update_idx, new_tgt_tokens, tgt_tokens)
                pred_tgt_tokens = tgt_tokens

            token_logits = torch.where(logits_update_idx, new_token_logits, token_logits)
            final_ffn = torch.where(ffn_update_idx, new_ffn, final_ffn)

        return token_logits, pred_tgt_tokens, alphas, final_ffn
