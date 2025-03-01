import torch
import torch.nn as nn
import json
from src.modules import Block, Embedding

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
