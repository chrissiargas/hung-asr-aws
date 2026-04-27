import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

from speechLM_utils.downsamplers import ReshapeAdapter, Conv1DAdapter, AvgPoolAdapter, LinearAdapter, CIFireAdapter


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class CrossAttention(nn.Module):
    def __init__(self,
                 hidden_dim,
                 audio_dim,
                 audio_offset=0,
                 num_heads=32,
                 causal_fusion: bool = False,
                 downsampler: Optional[nn.Module] = None,
                 downsample_L: int = 5,
                 gated: bool = False,
                 seq_len: int = 300,
                 positional_info: bool = False):

        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.gated = gated
        self.downsample_L = downsample_L
        self.positional_info = positional_info

        if self.head_dim * num_heads != hidden_dim:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        audio_embed_dim = audio_dim * self.downsample_L if isinstance(downsampler, ReshapeAdapter) else audio_dim

        self.k_proj = nn.Linear(audio_embed_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(audio_embed_dim, hidden_dim, bias=False)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(0.2)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.causal_fusion = causal_fusion
        self.audio_offset = audio_offset

        if self.gated:
            self.gate = nn.Parameter(torch.zeros(1))

        if self.positional_info:
            self.audio_pos_embed = SinusoidalPositionalEmbedding(d_model=audio_embed_dim, max_len=seq_len)

    def compute_mask(self, prm_len, prm_audio_len, inj_audio_len, device, dtype, prm_audio_mask=None, inj_audio_mask=None):
        min_val = torch.finfo(dtype).min

        if self.causal_fusion:
            batch_size = inj_audio_mask.shape[0]
            inj_audio_end = inj_audio_mask.sum(dim=1)
            inj_audio_masked = inj_audio_len - inj_audio_end
            prm_audio_end = prm_audio_mask.sum(dim=1)
            prm_audio_masked = prm_audio_len - prm_audio_end

            prm_indices = torch.arange(prm_len, device=device).view(1, prm_len, 1)
            inj_indices = torch.arange(inj_audio_len, device=device).view(1, 1 , inj_audio_len)

            content_prm_audio_len = prm_audio_len - prm_audio_masked
            content_inj_audio_len = inj_audio_len - inj_audio_masked

            slope = (content_prm_audio_len / content_inj_audio_len).view(batch_size, 1, 1)

            t_rel = torch.clamp(prm_indices - self.audio_offset, min=0)
            inj_boundary = torch.floor(t_rel * slope)

            causal_mask = torch.where(inj_indices <= inj_boundary, 0.0, min_val)
            causal_mask = causal_mask.unsqueeze(1)
        else:
            causal_mask = 0

        if inj_audio_mask is not None:
            audio_mask = inj_audio_mask[:, None, None, :]
            audio_mask = (1 - audio_mask) * min_val
        else:
            audio_mask = 0

        mask = causal_mask + audio_mask

        if isinstance(mask, torch.Tensor):
            mask = torch.clamp(mask, min=min_val).to(dtype)

        return mask

    def forward(self, hidden_states, audio_features, prompt_audio=None, inj_audio_mask=None, prm_audio_mask=None):
        batch_size, text_len, _ = hidden_states.shape
        inj_audio_len = audio_features.shape[1]
        prm_len = hidden_states.shape[1]

        if prompt_audio is not None:
            prm_audio_len = prompt_audio.shape[1]
        else:
            prm_audio_len = 0

        query = self.q_proj(self.layer_norm(hidden_states))

        if self.positional_info:
            audio_features = self.audio_pos_embed(audio_features)

        key = self.k_proj(audio_features)
        value = self.v_proj(audio_features)

        query = query.view(batch_size, text_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, inj_audio_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, inj_audio_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = self.compute_mask(prm_len, prm_audio_len, inj_audio_len, scores.device, scores.dtype, prm_audio_mask, inj_audio_mask)

        scores = scores + mask

        attentions = F.softmax(scores, dim=-1)
        attentions = self.dropout(attentions)

        attn_output = torch.matmul(attentions, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, text_len, self.hidden_dim)

        proj_audio_context = self.out_proj(attn_output)
        if self.gated:
            proj_audio_context = torch.tanh(self.gate) * proj_audio_context

        fused_output = hidden_states + proj_audio_context

        return fused_output

class InjectionLayer(nn.Module):
    def __init__(self, lm_layer, cross_attention_layer):
        super().__init__()

        self.LM_layer = lm_layer
        self.cross_attention_layer = cross_attention_layer

        self.injection_audio = None
        self.injection_audio_mask = None
        self.prompt_audio = None
        self.prompt_audio_mask = None

    def forward(self, hidden_states, *args, **kwargs):
        lm_outputs = self.LM_layer(hidden_states, *args, **kwargs)
        lm_hidden_states = lm_outputs[0]
            
        fused_hidden_states = self.cross_attention_layer(lm_hidden_states,
                                                           self.injection_audio,
                                                           self.prompt_audio,
                                                           self.injection_audio_mask,
                                                           self.prompt_audio_mask)

        rest = lm_outputs[1:] if len(lm_outputs) > 1 else ()
        if isinstance(rest, torch.Tensor):
            rest = (rest,)

        return (fused_hidden_states,) + rest
