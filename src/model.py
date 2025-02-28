import torch
import torch.nn as nn
import math
import json
from torch.nn import functional as F
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
            c_fc=nn.Linear(d_model, 4 * d_model),
            c_proj=nn.Linear(4 * d_model, d_model),
            act=NewGELU(),
            dropout=nn.Dropout(drop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlpf(self.ln_2(x))
        return x

class CCoTConfig:
    def __init__(
        self, 
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        num_encoder_layers: int = 3, 
        num_decoder_layers: int = 1, 
        encoder_hidden_size: int = 768,
        decoder_hidden_size: int = 768,
        vocab_size: int = 300,
        encoder_head: int = 12,
        decoder_head: int = 12,
        encoder_dropout: float = 0.1,
        decoder_dropout: float = 0.1,
        encoder_maxlen: int = 256,
        decoder_maxlen: int = 1024,
    ):
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.vocab_size = vocab_size
        self.encoder_head = encoder_head
        self.decoder_head = decoder_head
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout
        self.encoder_maxlen = encoder_maxlen
        self.decoder_maxlen = decoder_maxlen
    
    def load(json_file_path: str):
        with open(json_file_path, "r") as f:
            config = json.load(f)
        
        return CCoTConfig(**config)

class CCoTEncoder(nn.Module):
    def __init__(self, config: CCoTConfig):
        super().__init__()
        self.config = config
        self.encoderlayers = nn.ModuleList([Block(
            d_model        = config.encoder_hidden_size,
            nhead          = config.encoder_head,
            drop           = config.encoder_dropout,
            maxlen         = config.encoder_maxlen,
            rpe            = True,
            rpe_type       = 'rope',
            no_causal_mask = True, 
        ) for _ in range(config.num_encoder_layers)])
        self.ln_f = nn.LayerNorm(config.encoder_hidden_size)
        
    def forward(self, x):
        for layer in self.encoderlayers:
            x = layer(x)
        x = self.ln_f(x)
        return x

class CCoTDecoder(nn.Module):
    def __init__(self, config: CCoTConfig):
        super().__init__()
        self.config = config
        self.decoderlayers = nn.ModuleList([Block(
            d_model        = config.decoder_hidden_size,
            nhead          = config.decoder_head,
            drop           = config.decoder_dropout,
            maxlen         = config.decoder_maxlen,
            rpe            = True,
            rpe_type       = 'rope',
            no_causal_mask = True, 
        ) for _ in range(config.num_decoder_layers)])
        self.ln_f = nn.LayerNorm(config.decoder_hidden_size)
        
    def forward(self, x, mask=None):
        for layer in self.decoderlayers:
            x = layer(x, mask=mask)
        x = self.ln_f(x)
        return x

class CCoT(nn.Module):
    def __init__(self, args, logger, device):
        super().__init__()
        config = CCoTConfig.load(args.model_config)
        self.args = args
        self.logger = logger
        self.config = config

        if args.rank == 0:
            cfg_json = json.dumps(config.__dict__, indent=2)
            logger.info(f"Model configuration: {cfg_json}")

        self.encoder_embed = Embedding(d_model=config.encoder_hidden_size, vocab_size=config.vocab_size)
        self.decoder_embed = Embedding(d_model=config.decoder_hidden_size, vocab_size=config.vocab_size)
        self.norm = nn.LayerNorm(config.encoder_hidden_size)
        self.intermediate = nn.Linear(config.encoder_hidden_size, config.decoder_hidden_size)
        self.lm_head = nn.Linear(config.decoder_hidden_size, config.vocab_size, bias=True)

        self.encoder = CCoTEncoder(config)
        self.decoder = CCoTDecoder(config)

        self.device = device
        self.to(device)

        if hasattr(args, "model_path"):
            self.load(args.model_path)

    def load(self, model_path: str):
        self.logger.info(f"Loading model from {model_path}...")
        state_dict = torch.load(model_path, map_location=self.device)

        new_state_dict = self.state_dict()

        for key in state_dict.keys():  
            if 'RPE' not in key:
                new_state_dict[key] = state_dict[key]

        self.load_state_dict(new_state_dict, strict=True)
        self.logger.info(f"Model loaded from {model_path}")
    
    def encode(
        self, 
        input_ids: torch.Tensor,
        num_loops: int = 0,
        num_contemplation_tokens: int = 0
    ):
        input_embeds = self.encoder_embed(input_ids)
        hidden_states = input_embeds
        bz, dim = hidden_states.shape[0], hidden_states.shape[2]

        hidden_states = torch.cat([hidden_states, torch.zeros(bz, num_contemplation_tokens, dim).to(input_embeds.device)], dim=1)

        residual = hidden_states

        for iteration in range(num_loops):
            hidden_states = self.encoder(hidden_states)
            hidden_states += residual
        
        hidden_states = self.norm(hidden_states)
        return hidden_states

    def decode(
        self, 
        hidden_states: torch.Tensor,
        answer_ids: torch.Tensor,
    ):
        thoughts_len = hidden_states.shape[1]
        answer_embeds = self.decoder_embed(answer_ids)
        final_seq = torch.cat([hidden_states, answer_embeds], dim=1)
        seq_len = final_seq.shape[1]
        
        mask = torch.tril(torch.ones(seq_len, seq_len))
        mask[:, :thoughts_len] = 1
        mask = mask.view(1, 1, seq_len, seq_len).to(final_seq.device)

        logits = self.lm_head(self.decoder(final_seq, mask=mask))
        return logits

    def forward(
        self, 
        input_ids: torch.Tensor,
        answer_ids: torch.Tensor,
        num_loops: int = 0, 
        num_contemplation_tokens: int = 0
    ):
        hidden_states = self.encode(input_ids, num_loops, num_contemplation_tokens)
        hidden_states = self.intermediate(hidden_states)
        return self.decode(hidden_states, answer_ids)
    
    def generate(
        self, 
        input_ids: torch.Tensor,
        max_length: int = 200,
        num_loops: int = 0,
        num_contemplation_tokens: int = 0
    ):
        hidden_states = self.encode(input_ids, num_loops, num_contemplation_tokens)
        hidden_states = self.intermediate(hidden_states)

        shift = hidden_states.shape[1]

        bz = input_ids.shape[0]

        answer = torch.zeros((bz, max_length), dtype=torch.long).to(input_ids.device)
        answer[:, 0] = self.config.bos_token_id
        cur = torch.zeros(bz, dtype=torch.long).to(input_ids.device)

        for _ in range(max_length - 1):
            logits = self.decode(hidden_states, answer)
            idx_new = torch.argmax(logits, dim=2).int()[torch.arange(bz), cur + shift]
            answer[torch.arange(bz), cur + 1] = idx_new.long()

            if torch.sum(idx_new == self.config.eos_token_id) == bz:
                break

            cur[idx_new != self.config.eos_token_id] += 1
        
        return answer, answer[torch.arange(bz), cur]

