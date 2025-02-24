import torch
import json
import torch.nn as nn
from torch.nn import functional as F

from transformers import Qwen2Config, AutoModelForCausalLM
from transformers.cache_utils import Cache
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2RotaryEmbedding, Qwen2Attention, Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2ForCausalLM
)
from typing import Optional, Tuple

class CCoTConfig(Qwen2Config):
    def __init__(
        self, 
        num_encoder_layers: int = 3, 
        num_loop_layers: int = 4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_encoder_layers = num_encoder_layers
        self.num_loop_layers = num_loop_layers
        self.model_path = "Qwen/Qwen2.5-1.5B-Instruct"
    
    def load(json_file_path: str):
        with open(json_file_path, "r") as f:
            config = json.load(f)
        
        return CCoTConfig(**config)

class Attention(Qwen2Attention):
    def __init__(self, is_causal: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.is_causal = is_causal

class Block(Qwen2DecoderLayer):
    def __init__(self, config: CCoTConfig, layer_idx: int, is_causal: bool = True):
        super().__init__(config=config, layer_idx=layer_idx)
        self.self_attn = Attention(config=config, layer_idx=layer_idx, is_causal=is_causal)

class CCoT(nn.Module):
    def __init__(self, config: CCoTConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.encoderlayers = nn.ModuleList(
            [Block(config, layer_idx, False) for layer_idx in range(config.num_encoder_layers)]
        )
        self.decoderlayers = nn.ModuleList(
            [Block(config, layer_idx, True) for layer_idx in range(config.num_hidden_layers)]
        )

        self.load_from_pretrained_standard_cot(self.config.model_path)

    def get_position_embeds(self, x):
        length = x.shape[1]
        position_ids = torch.arange(length, device=x.device).unsqueeze(0)
        return self.rotary_emb(x, position_ids)

    def forward(
        self, 
        input_ids: torch.Tensor,
        answer_ids: torch.Tensor,
        num_loops: int = 0, 
        num_contemplation_tokens: int = 0
    ):
        input_embeds = self.embed_tokens(input_ids)
        answer_embeds = self.embed_tokens(answer_ids)
        
        hidden_states = input_embeds
        bz, dim = hidden_states.shape[0], hidden_states.shape[2]


        hidden_states = torch.cat([hidden_states, torch.zeros(bz, num_contemplation_tokens, dim).to(input_embeds.device)], dim=1)
        position_embeddings = self.get_position_embeds(hidden_states)

        residual = hidden_states

        for iteration in range(num_loops):
            for encoderlayer in self.encoderlayers:
                hidden_states = encoderlayer(hidden_states, position_embeddings)
            hidden_states += residual

        hidden_states = self.norm(hidden_states)
        
        hidden_states = torch.cat([hidden_states, answer_embeds], dim=1)
        position_embeddings = self.get_position_embeds(hidden_states)

        for decoderlayer in self.decoderlayers:
            hidden_states = decoderlayer(hidden_states, position_embeddings)
        
        logits = self.lm_head(hidden_states)
        return logits

    def load_from_pretrained_standard_cot(self, model_path: str):
        model = AutoModelForCausalLM.from_pretrained(model_path, tie_word_embeddings=False)

        target_state_dict = model.state_dict()
        new_state_dict = self.state_dict()

        replacements = {
            "layers": "decoderlayers", 
            "model.": ""
        }
        
        load_params = []
        for key in target_state_dict.keys():
            name = key
            for old, new in replacements.items():
                name = name.replace(old, new)
            if name in new_state_dict:
                new_state_dict[name] = target_state_dict[key]
                load_params.append(name)
        
        print(f"{load_params=}")
        
        self.load_state_dict(new_state_dict)

    def generate(
        self, 
        input_ids: torch.Tensor,
        max_length: int = 100,
        num_loops: int = 0,
        num_contemplation_tokens: int = 0
    ):
        input_embeds = self.embed_tokens(input_ids)
        pass