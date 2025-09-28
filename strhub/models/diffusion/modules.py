import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import transformer
from timm.models.vision_transformer import VisionTransformer, PatchEmbed


class SinusoidalPosEmb(nn.Module):
    def __init__(self, num_steps, dim, rescale_steps=8000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, time):
        time = time / self.num_steps * self.rescale_steps
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class AdaLayerNorm(nn.Module):
    def __init__(self, n_embd, diffusion_step, eps=1e-5):
        super().__init__()
        self.emb = SinusoidalPosEmb(diffusion_step, n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd * 2)
        self.layernorm = nn.LayerNorm(n_embd, eps=eps)

    def forward(self, x, timestep):
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x


class DecoderLayerWot(nn.Module):
    """
    Parallel Decoder
    """
    def __init__(self, d_model, nhead, time_step, dim_feedforward=2048, dropout=0.1, activation='relu',
                 layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)

        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear3 = nn.Linear(d_model, dim_feedforward)
        self.linear4 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm4 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = transformer._get_activation_fn(activation)

    def forward(self, predicts_embed, feature_map, key_padding_mask, attn_mask):
        # mask_self_attention
        outputs, _ = self.self_attn(predicts_embed, predicts_embed, predicts_embed,
                                    key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # print(outputs)
        outputs = self.norm1(predicts_embed + self.dropout1(outputs))
        # ff
        self_attn_outputs = self.norm2(outputs + self.linear2(self.activation(self.linear1(outputs))))

        outputs2, alphas = self.cross_attn(self_attn_outputs, feature_map, feature_map)
        outputs = self.norm3(self_attn_outputs + self.dropout2(outputs2))
        # ff
        cross_attn_outputs = self.norm4(outputs + self.linear4(self.activation(self.linear3(outputs))))
        return cross_attn_outputs, alphas


class DecoderLayerEdit(nn.Module):
    """
    Parallel Decoder
    """
    def __init__(self, d_model, nhead, time_step, dim_feedforward=2048, dropout=0.1, activation='gelu',
                 layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = AdaLayerNorm(d_model, time_step, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)

        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = AdaLayerNorm(d_model, time_step, eps=layer_norm_eps)
        self.dropout2 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = AdaLayerNorm(d_model, time_step, eps=layer_norm_eps)
        self.dropout4 = nn.Dropout(dropout)
        self.activation = transformer._get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, t, predicts_embed, feature_map, key_padding_mask, attn_mask, uncondition=False):
        tgt2, attn = self.self_attn(predicts_embed, predicts_embed, predicts_embed,
                                    key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        tgt = predicts_embed + self.dropout1(tgt2)
        tgt = self.norm1(tgt, t)
        
        if uncondition:
            return tgt, attn
        else:
            tgt2, attn2 = self.cross_attn(tgt, feature_map, feature_map)
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt, t)
            
            tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout4(tgt2)
            tgt = self.norm3(tgt, t)
            
        return tgt, attn


class DecoderLayer(nn.Module):
    """
    Parallel Decoder
    """
    def __init__(self, d_model, nhead, time_step, dim_feedforward=2048, dropout=0.1, activation='relu',
                 layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = AdaLayerNorm(d_model, time_step, eps=layer_norm_eps)
        self.norm2 = AdaLayerNorm(d_model, time_step, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear3 = nn.Linear(d_model, dim_feedforward)
        self.linear4 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = AdaLayerNorm(d_model, time_step, eps=layer_norm_eps)
        self.norm4 = AdaLayerNorm(d_model, time_step, eps=layer_norm_eps)
        # self.dropout3 = nn.Dropout(dropout)
        # self.dropout4 = nn.Dropout(dropout)

        self.activation = transformer._get_activation_fn(activation)

    def forward(self, t, predicts_embed, feature_map, key_padding_mask, attn_mask):
        # mask_self_attention
        outputs, _ = self.self_attn(predicts_embed, predicts_embed, predicts_embed,
                                    key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # print(outputs)
        outputs = self.norm1(predicts_embed + self.dropout1(outputs), t)
        # ff
        # self_attn_outputs = self.norm2(outputs + self.dropout2(self.linear2(self.activation(self.linear1(outputs)))), t)
        self_attn_outputs = self.norm2(outputs + self.linear2(self.activation(self.linear1(outputs))), t)

        outputs2, alphas = self.cross_attn(self_attn_outputs, feature_map, feature_map)
        outputs = self.norm3(self_attn_outputs + self.dropout2(outputs2), t)
        # ff
        # cross_attn_outputs = self.norm4(outputs + self.dropout4(self.linear4(self.activation(self.linear3(outputs)))),
        #                                 t)
        cross_attn_outputs = self.norm4(outputs + self.linear4(self.activation(self.linear3(outputs))), t)
        return cross_attn_outputs, alphas


class DecoderLayerRevert(nn.Module):
    """
    Parallel Decoder
    """
    def __init__(self, d_model, nhead, time_step, dim_feedforward=2048, dropout=0.1, activation='relu',
                 layer_norm_eps=1e-5):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = AdaLayerNorm(d_model, time_step, eps=layer_norm_eps)
        self.norm2 = AdaLayerNorm(d_model, time_step, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear3 = nn.Linear(d_model, dim_feedforward)
        self.linear4 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = AdaLayerNorm(d_model, time_step, eps=layer_norm_eps)
        self.norm4 = AdaLayerNorm(d_model, time_step, eps=layer_norm_eps)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = transformer._get_activation_fn(activation)

    def forward(self, t, predicts_embed, feature_map, key_padding_mask, attn_mask):
        # mask_self_attention
        outputs, _ = self.cross_attn(predicts_embed, feature_map, feature_map)
        # print(outputs)
        outputs = self.norm1(predicts_embed + self.dropout1(outputs), t)
        # ff
        outputs = self.norm2(outputs + self.linear2(self.activation(self.linear1(outputs))), t)

        outputs2, alphas = self.self_attn(outputs, outputs, outputs,
                                          key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        outputs = self.norm3(outputs + self.dropout2(outputs2), t)
        # ff
        outputs = self.norm4(outputs + self.linear4(self.activation(self.linear3(outputs))), t)
        return outputs, alphas


class DecoderLayerCross(nn.Module):
    """
    Parallel Decoder
    """
    def __init__(self, d_model, nhead, time_step, dim_feedforward=2048, dropout=0.1, activation='relu',
                 layer_norm_eps=1e-5):
        super().__init__()
        self.cross_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = AdaLayerNorm(d_model, time_step, eps=layer_norm_eps)
        self.norm2 = AdaLayerNorm(d_model, time_step, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)

        self.cross_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear3 = nn.Linear(d_model, dim_feedforward)
        self.linear4 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = AdaLayerNorm(d_model, time_step, eps=layer_norm_eps)
        self.norm4 = AdaLayerNorm(d_model, time_step, eps=layer_norm_eps)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = transformer._get_activation_fn(activation)

    def forward(self, t, pos_embed, predicts_embed, feature_map, key_padding_mask, attn_mask):
        # mask_self_attention
        outputs, _ = self.cross_attn1(pos_embed, predicts_embed, predicts_embed,
                                      key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # print(outputs)
        outputs = self.norm1(pos_embed + self.dropout1(outputs), t)
        # ff
        attn_outputs = self.norm2(outputs + self.linear2(self.activation(self.linear1(outputs))), t)

        outputs2, alphas = self.cross_attn2(attn_outputs, feature_map, feature_map)
        outputs = self.norm3(attn_outputs + self.dropout2(outputs2), t)
        # ff
        cross_attn_outputs = self.norm4(outputs + self.linear4(self.activation(self.linear3(outputs))), t)
        return cross_attn_outputs, alphas


class DecoderCross(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm):
        super().__init__()
        self.layers = transformer._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, t, pos_query, tgt_token, feature_map, key_padding_mask, attn_mask):
        for i, mod in enumerate(self.layers):
            predicts, alphas = mod(t, pos_query, tgt_token, feature_map, key_padding_mask, attn_mask)
        return predicts


class DecoderWot(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm):
        super().__init__()
        self.layers = transformer._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt_token, feature_map, key_padding_mask, attn_mask):
        for i, mod in enumerate(self.layers):
            predicts, alphas = mod(tgt_token, feature_map, key_padding_mask, attn_mask)
        return predicts


class Decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm):
        super().__init__()
        self.layers = transformer._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, t, tgt_token, feature_map, key_padding_mask, attn_mask, uncondition=False):
        for i, mod in enumerate(self.layers):
            predicts, alphas = mod(t, tgt_token, feature_map, key_padding_mask, attn_mask, uncondition)
        return predicts


class TokenEmbedding(nn.Module):

    def __init__(self, charset_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(charset_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, tokens: torch.Tensor):
        return math.sqrt(self.embed_dim) * self.embedding(tokens)


class Encoder(VisionTransformer):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 class_token=False, qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 embed_layer=PatchEmbed):
        super().__init__(img_size, patch_size, in_chans, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                         mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=drop_path_rate, embed_layer=embed_layer,
                         num_classes=0, global_pool='', class_token=class_token)  # these disable the classifier head

        def forward(self, x):
            # Return all tokens
            return self.forward_features(x)
        
    def load_pretrained_weights(self, path):
        pretrained_dict = torch.load(path, map_location='cpu')
        model_dict = self.state_dict()
        # Filter out unnecessary keys from pretrained state dict
        filtered_dict = {}
        for k, v in pretrained_dict['model'].items():
            if k in model_dict:
                if k=='pos_embed':
                    filtered_dict[k] = v[:,1:,:]
                else:
                    filtered_dict[k] = v
        # pretrained_dict = {k: v for k, v in pretrained_dict['model'].items() if k in model_dict}
        print("filtered_dict:", filtered_dict.keys())
        
        # Update the state dict
        model_dict.update(filtered_dict)
        self.load_state_dict(model_dict)


