# -*- coding: utf-8 -*-
# Copyright (c) 2025 Meituan
# This code is licensed under the MIT License, for details, see the ./LICENSE file. 

from typing import Optional, Tuple, Dict, List

import torch
from torch import nn

from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import auto_docstring, logging
from transformers.models.longcat_flash.modeling_longcat_flash import (
    LongcatFlashForCausalLM,
    LongcatFlashModel,
    LongcatFlashRMSNorm,
    LongcatFlashRotaryEmbedding,
    LongcatFlashDecoderLayer,
    LongcatFlashPreTrainedModel,
)
from .configuration_longcat_ngram import LongcatFlashNgramConfig

logger = logging.get_logger(__name__)


@auto_docstring
class LongcatFlashNgramPreTrainedModel(LongcatFlashPreTrainedModel):
    pass


class NgramCache(DynamicCache):
    """
    Extended DynamicCache for storing N-gram context alongside KV cache.
    """
    def __init__(self, config=None):
        super().__init__()
        self.ngram_context = None
        # Keep only n-1 tokens (minimum needed for N-gram computation)
        self.max_context_len = config.emb_neighbor_num - 1
    
    def update_ngram_context(self, new_tokens: torch.Tensor) -> None:
        """
        Update N-gram context with window management.
        
        Args:
            new_tokens: New tokens to append, shape (batch_size, seq_len)
        """
        if self.ngram_context is None:
            self.ngram_context = new_tokens.clone()
        else:
            self.ngram_context = torch.cat([self.ngram_context, new_tokens], dim=-1)
        
        # Truncate to maintain constant memory footprint
        if self.ngram_context.size(-1) > self.max_context_len:
            self.ngram_context = self.ngram_context[..., -self.max_context_len:]
    
    def reorder_cache(self, beam_idx: torch.LongTensor) -> "Cache":
        """Reorder cache for beam search."""
        # Reorder parent's KV cache
        super().reorder_cache(beam_idx)
        
        # Reorder N-gram context
        if self.ngram_context is not None:
            self.ngram_context = self.ngram_context.index_select(0, beam_idx.to(self.ngram_context.device))
        
        return self


class NgramEmbedding(nn.Module):
    """
    Computes embeddings enriched with N-gram features without maintaining internal state.
    """
    def __init__(self, config, base_embeddings):
        super().__init__()
        self.config = config
        self.word_embeddings = base_embeddings
        
        self.m = config.ngram_vocab_size_ratio * config.vocab_size
        self.k = config.emb_split_num
        self.n = config.emb_neighbor_num
        
        self._init_ngram_embeddings()
        self._vocab_mods_cache = None
        
    def _init_ngram_embeddings(self) -> None:
        """Initialize N-gram embedding and projection layers."""
        num_embedders = self.k * (self.n - 1)
        emb_dim = self.config.hidden_size // num_embedders
        
        embedders = []
        post_projs = []
        
        for i in range(num_embedders):
            vocab_size = int(self.m + i * 2 + 1)
            emb = nn.Embedding(vocab_size, emb_dim, padding_idx=self.config.pad_token_id)
            proj = nn.Linear(emb_dim, self.config.hidden_size, bias=False)
            embedders.append(emb)
            post_projs.append(proj)
        
        self.embedders = nn.ModuleList(embedders)
        self.post_projs = nn.ModuleList(post_projs)
    
    def _shift_right_ignore_eos(self, tensor: torch.Tensor, n: int, eos_token_id: int = 2) -> torch.Tensor:
        """Shift tensor right by n positions, resetting at EOS tokens."""
        batch_size, seq_len = tensor.shape
        result = torch.zeros_like(tensor)
        eos_mask = (tensor == eos_token_id)
        
        for i in range(batch_size):
            eos_positions = eos_mask[i].nonzero(as_tuple=True)[0]
            prev_idx = 0
            
            for eos_idx in eos_positions:
                end_idx = eos_idx.item() + 1
                if end_idx - prev_idx > n:
                    result[i, prev_idx+n:end_idx] = tensor[i, prev_idx:end_idx-n]
                prev_idx = end_idx
            
            if prev_idx < seq_len and seq_len - prev_idx > n:
                result[i, prev_idx+n:seq_len] = tensor[i, prev_idx:seq_len-n]
        
        return result
    
    def _precompute_vocab_mods(self) -> Dict[Tuple[int, int], List[int]]:
        """Precompute modular arithmetic values for vocabulary."""
        if self._vocab_mods_cache is not None:
            return self._vocab_mods_cache
        
        vocab_mods = {}
        vocab_size = self.config.vocab_size
        
        for i in range(2, self.n + 1):
            for j in range(self.k):
                index = (i - 2) * self.k + j
                emb_vocab_dim = int(self.m + index * 2 + 1)
                
                mods = []
                power_mod = 1
                for _ in range(i - 1):
                    power_mod = (power_mod * vocab_size) % emb_vocab_dim
                    mods.append(power_mod)
                
                vocab_mods[(i, j)] = mods
        
        self._vocab_mods_cache = vocab_mods
        return vocab_mods
    
    def _get_ngram_ids(
        self, 
        input_ids: torch.Tensor, 
        shifted_ids: Dict[int, torch.Tensor], 
        vocab_mods: List[int], 
        ngram: int
    ) -> torch.Tensor:
        """Compute N-gram hash IDs using polynomial rolling hash."""
        ngram_ids = input_ids.clone()
        for k in range(2, ngram + 1):
            ngram_ids = ngram_ids + shifted_ids[k] * vocab_mods[k - 2]
        return ngram_ids
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        ngram_context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Stateless forward pass.
        
        Args:
            input_ids: Current input token IDs of shape (batch_size, seq_len)
            ngram_context: Optional historical context of shape (batch_size, context_len)
            
        Returns:
            Embedding tensor of shape (batch_size, seq_len, hidden_size)
        """
        seq_len = input_ids.size(-1)
        
        # Determine complete context
        if ngram_context is not None:
            context = torch.cat([ngram_context[..., -(self.n-1):], input_ids], dim=-1)
        else:
            context = input_ids
        
        # Base word embeddings
        device = self.word_embeddings.weight.device
        x = self.word_embeddings(input_ids.to(device)).clone()
        
        # Precompute modular values
        vocab_mods = self._precompute_vocab_mods()
        
        # Compute shifted IDs
        shifted_ids = {}
        for i in range(2, self.n + 1):
            shifted_ids[i] = self._shift_right_ignore_eos(
                context, i - 1, eos_token_id=self.config.eos_token_id
            )
        
        # Add N-gram embeddings
        for i in range(2, self.n + 1):
            for j in range(self.k):
                index = (i - 2) * self.k + j
                emb_vocab_dim = int(self.m + index * 2 + 1)
                
                ngram_ids = self._get_ngram_ids(context, shifted_ids, vocab_mods[(i, j)], ngram=i)
                new_ids = (ngram_ids % emb_vocab_dim)[..., -seq_len:]
                
                embedder_device = self.embedders[index].weight.device
                x_ngram = self.embedders[index](new_ids.to(embedder_device))
                
                proj_device = self.post_projs[index].weight.device
                x_proj = self.post_projs[index](x_ngram.to(proj_device))
                x = x + x_proj.to(x.device)
        
        # Normalize
        x = x / (1 + self.k * (self.n - 1))
        
        return x


class LongcatFlashNgramModel(LongcatFlashModel):
    """LongcatFlash model with N-gram enhanced embeddings."""
    _keys_to_ignore_on_load_unexpected = [r"model\.mtp.*"]
    config_class = LongcatFlashNgramConfig
    
    def __init__(self, config):
        super().__init__(config)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.ngram_embeddings = NgramEmbedding(config, self.embed_tokens)

        self.layers = nn.ModuleList(
            [LongcatFlashDecoderLayer(config, layer_idx) for layer_idx in range(config.num_layers)]
        )
        
        self.head_dim = config.head_dim  
        self.config.num_hidden_layers = 2 * config.num_layers
        self.norm = LongcatFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LongcatFlashRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # Extract N-gram context if available
        ngram_context = None
        if isinstance(past_key_values, NgramCache) and past_key_values.ngram_context is not None:
            ngram_context = past_key_values.ngram_context

        if inputs_embeds is None:
            inputs_embeds = self.ngram_embeddings(input_ids, ngram_context=ngram_context)

        # Initialize NgramCache if needed
        if use_cache and past_key_values is None:
            past_key_values = NgramCache(config=self.config)
        
        # Update N-gram context
        if use_cache and isinstance(past_key_values, NgramCache):
            past_key_values.update_ngram_context(input_ids)

        # Prepare cache position
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                inputs_embeds.shape[1], device=inputs_embeds.device
            ) + past_seen_tokens

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Create causal mask
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        # Forward through decoder layers
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )


class LongcatFlashNgramForCausalLM(LongcatFlashForCausalLM):
    """LongcatFlash model for causal language modeling with N-gram embeddings."""
    _keys_to_ignore_on_load_unexpected = [r"model\.mtp.*"]
    config_class = LongcatFlashNgramConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = LongcatFlashNgramModel(config)

    @torch.no_grad()
    def generate(self, inputs=None, generation_config=None, **kwargs):
        """Override to ensure NgramCache is used."""

        if "past_key_values" not in kwargs or kwargs["past_key_values"] is None:
            kwargs["past_key_values"] = NgramCache(config=self.config)
        
        return super().generate(inputs=inputs, generation_config=generation_config, **kwargs)

__all__ = ["LongcatFlashNgramPreTrainedModel", "LongcatFlashNgramModel", "LongcatFlashNgramForCausalLM"]