import torch
import torch.nn as nn
import math
import xformers.ops as xops 


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(
                seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )
    
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin) if k is not None else None
    return q_embed, k_embed

class Embedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        embedding = self.tok_embed(x)
        return self.norm(embedding)

class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, drop, maxlen, rpe, rpe_type, no_causal_mask):
        super().__init__()
        assert d_model % nhead == 0
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(drop)
        self.resid_dropout = nn.Dropout(drop)

        self.bias = torch.tril(torch.ones(maxlen, maxlen)).view(1, 1, maxlen, maxlen)   
        self.n_head = nhead
        self.n_embd = d_model

        self.rpe = rpe
        self.rpe_type = rpe_type
        self.no_causal_mask = no_causal_mask

        if self.rpe:
            if self.rpe_type == "rope":
                self.rotary_emb = LlamaRotaryEmbedding(
                    self.n_embd // self.n_head, max_position_embeddings=maxlen
                )
            else:
                raise NotImplementedError

    def forward(self, x, mask=None):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.rpe and self.rpe_type == "rope":
            position_ids = torch.arange(
                0, T, dtype=torch.long, device=x.device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, T)
            cos, sin = self.rotary_emb(v, seq_len=T)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        if self.rpe and self.rpe_type == 'alibi':
            raise NotImplementedError
        attn_bias = None

        if not self.no_causal_mask:
            attn_bias = xops.LowerTriangularMask()

        if mask is not None:
            custom_mask = mask.to(dtype=q.dtype, device=q.device).masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0)
            aligned_T = (T + 7) // 8 * 8
            aligned_attn_bias = torch.zeros((B, self.n_head, T, aligned_T), dtype=q.dtype, device=q.device)
            aligned_attn_bias[..., :T] = custom_mask.expand(B, self.n_head, T, T)
            custom_mask = aligned_attn_bias[..., :T]

            if attn_bias is None:
                attn_bias = custom_mask
            else:
                attn_bias = attn_bias.add_bias(custom_mask)

        y = xops.memory_efficient_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_bias=attn_bias,
        ).reshape(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    def __init__(self, d_model, nhead, drop, maxlen, rpe, rpe_type, no_causal_mask):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model, nhead, drop, maxlen, rpe, rpe_type, no_causal_mask
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(d_model, 4 * d_model),
            c_proj  = nn.Linear(4 * d_model, d_model),
            act     = NewGELU(),
            dropout = nn.Dropout(drop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlpf(self.ln_2(x))
        return x