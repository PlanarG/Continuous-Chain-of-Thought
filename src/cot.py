import torch
import torch.nn as nn
import json
from src.modules import Block, Embedding

class CoTConfig:
    def __init__(
        self, 
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        num_layers: int = 3,
        hidden_size: int = 768,
        vocab_size: int = 300,
        num_heads: int = 12,
        dropout: float = 0.1,
        maxlen: int = 1024,
    ):
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.num_layers   = num_layers
        self.hidden_size  = hidden_size
        self.vocab_size   = vocab_size
        self.num_heads    = num_heads
        self.dropout      = dropout
        self.maxlen       = maxlen
    
    def load(json_file_path: str):
        with open(json_file_path, "r") as f:
            config = json.load(f)
        
        return CoTConfig(**config)
    
class CoTDecoder(nn.Module):
    def __init__(self, config: CoTConfig):
        super().__init__()
        self.config = config
        self.decoder_layers = nn.ModuleList([Block(
            d_model        = config.hidden_size,
            nhead          = config.num_heads,
            drop           = config.dropout,
            maxlen         = config.maxlen,
            rpe            = True,
            rpe_type       = 'rope',
            no_causal_mask = False, 
        ) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size)
        
    def forward(self, x):
        for layer in self.decoder_layers:
            x = layer(x)
        x = self.ln_f(x)
        return x

class CoT(nn.Module):
    def __init__(self, args, logger, device):
        super().__init__()
        self.config = CoTConfig.load(args.model_config)
        self.args = args
        self.logger = logger

        if args.rank == 0:
            cfg_json = json.dumps(self.config.__dict__, indent=2)
            logger.info(f"Model configuration: {cfg_json}")
        
        self.embed = Embedding(self.config.hidden_size, self.config.vocab_size)
        self.decoder = CoTDecoder(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=True)

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
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        answer_ids: torch.Tensor,
        num_loops: int = 0, 
        num_contemplation_tokens: int = 0
    ):
        input = torch.cat([input_ids, answer_ids], dim=1)
        input = self.embed(input)
        logits = self.lm_head(self.decoder(input))
        return logits
    
    def generate(
        self, 
        input_ids: torch.Tensor,
        max_length: int = 200,
        num_loops: int = 0,
        num_contemplation_tokens: int = 0
    ):
        hidden_states = self.embed(input_ids)

        bz, shift = hidden_states.shape[0], hidden_states.shape[1]

        answer = torch.zeros((bz, max_length), dtype=torch.long).to(input_ids.device)
        answer[:, 0] = self.config.bos_token_id
        cur = torch.zeros(bz, dtype=torch.long).to(input_ids.device)

        for _ in range(max_length - 1):
            logits = self.forward(input_ids, answer)
            idx_new = torch.argmax(logits, dim=2).int()[torch.arange(bz), cur + shift]
            answer[torch.arange(bz), cur + 1] = idx_new.long()

            if torch.sum(idx_new == self.config.eos_token_id) == bz:
                break

            cur[idx_new != self.config.eos_token_id] += 1
        
        return answer, answer[torch.arange(bz), cur]

        
