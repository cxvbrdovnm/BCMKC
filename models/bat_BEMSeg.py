import torch.nn.functional as F
from torch import nn

class BAT(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, nhead=8, dropout=0.0, activation="relu"):
        super().__init__()

        # self.self_attn_text = MultiHeadSelfAttention(d_model=d_model, nhead=nhead, dropout=dropout)
        # self.self_attn_img = MultiHeadSelfAttention(d_model=d_model, nhead=nhead, dropout=dropout)
        self.cross_attn_text = MultiHeadSelfAttention(d_model=d_model, nhead=nhead, dropout=dropout)
        self.cross_attn_img = MultiHeadSelfAttention(d_model=d_model, nhead=nhead, dropout=dropout)

        self.ffn_text = ffn(d_model=d_model, d_ffn=d_ffn, dropout=dropout, activation=activation)
        self.ffn_img = ffn(d_model=d_model, d_ffn=d_ffn, dropout=dropout, activation=activation)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, visual, visual_pos, visual_masks, text, text_pos, text_masks):
        visual_q = self.with_pos_embed(visual, visual_pos)
        text_q = self.with_pos_embed(text, text_pos)
        # visual_self = self.self_attn_img(visual_q, visual_q, visual, visual_masks)
        # text_self = self.self_attn_text(text_q, text_q, text, text_masks)

        visual_cross = self.cross_attn_img(visual_q, text_q, text, text_masks)
        text_cross = self.cross_attn_text(text_q, visual_q, visual, visual_masks)

        visual_feat = self.ffn_img(visual_cross)
        text_feat = self.ffn_text(text_cross)

        return visual_feat, text_feat

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, q, k, v, key_padding_mask=None, attn_mask=None):
        src2 = self.self_attn(query=q, key=k, value=v,
                              attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]
        src = q + self.dropout1(src2)
        src = self.norm1(src)
        return src



class ffn(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.0, activation="relu"):
        super().__init__()
        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")