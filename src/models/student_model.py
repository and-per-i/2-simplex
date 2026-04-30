"""
StudentForCausalLM — Modello Student custom compatibile con HuggingFace Trainer.

Architettura: Decoder-only Transformer (Causal LM) con:
- Token embeddings + Positional embeddings (learnable)
- N layer Transformer decoder con causal self-attention
- LayerNorm finale
- LM head (linear projection → vocab_size logits)

Il modello è progettato per essere sostituibile con 2-Simplex attention
(impostare config.use_simplex_attention = True quando il kernel Triton è pronto).

Contratto HuggingFace rispettato:
- Eredita da PreTrainedModel
- Restituisce CausalLMOutputWithPast (ha .logits e .loss)
- Implementa _init_weights per inizializzazione standard HF
- Compatible con .save_pretrained() / .from_pretrained()
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .student_config import StudentConfig
from .two_simplicial_attention import TwoSimplicialAttention


# ---------------------------------------------------------------------------
# Blocchi costitutivi
# ---------------------------------------------------------------------------


class StudentEmbeddings(nn.Module):
    """Token embeddings + positional embeddings learnable."""

    def __init__(self, config: StudentConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S = input_ids.shape
        device = input_ids.device

        if attention_mask is not None:
            # Calcola posizioni incrementali ignorando il padding
            # [0, 0, 1, 1, 1] -> [0, 0, 0, 1, 2]
            position_ids = (torch.cumsum(attention_mask, dim=1) - 1).clamp(min=0)
        else:
            position_ids = torch.arange(S, dtype=torch.long, device=device).unsqueeze(0).expand(B, S)

        token_emb = self.token_embedding(input_ids)           # (B, S, H)
        pos_emb = self.position_embedding(position_ids)       # (B, S, H)

        return self.dropout(token_emb + pos_emb)


class StudentAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention standard.

    Placeholder per 2-Simplex attention: quando config.use_simplex_attention=True
    e il kernel Triton sarà disponibile, questa classe verrà sostituita.
    """

    def __init__(self, config: StudentConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.attn_dropout = nn.Dropout(config.dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, _ = hidden_states.shape

        def split_heads(x: torch.Tensor) -> torch.Tensor:
            # (B, S, H) → (B, num_heads, S, head_dim)
            return x.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        Q = split_heads(self.q_proj(hidden_states))
        K = split_heads(self.k_proj(hidden_states))
        V = split_heads(self.v_proj(hidden_states))

        # Converte la maschera HF (1=valido, 0=padding, dtype=long) in formato
        # additivo float richiesto da scaled_dot_product_attention:
        # 0.0 sulle posizioni valide, -inf sulle posizioni mascherate (padding).
        sdpa_mask = None
        is_causal = True
        if attention_mask is not None:
            # (B, S) → (B, 1, 1, S) per broadcasting su (B, heads, S_q, S_k)
            pad_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)  # True=valido
            sdpa_mask = torch.zeros(B, 1, S, S, dtype=Q.dtype, device=Q.device)
            sdpa_mask = sdpa_mask.masked_fill(~pad_mask, float("-inf"))
            # Aggiungi la maschera causale manualmente
            causal = torch.triu(
                torch.full((S, S), float("-inf"), device=Q.device, dtype=Q.dtype),
                diagonal=1,
            )
            sdpa_mask = sdpa_mask + causal
            is_causal = False  # la maschera causale è già incorporata in sdpa_mask

        # Scaled dot-product — usa Flash Attention se disponibile (PyTorch ≥ 2.0)
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=sdpa_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=is_causal,
        )

        # (B, num_heads, S, head_dim) → (B, S, H)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.hidden_size)
        return self.out_proj(attn_out)


class StudentFeedForward(nn.Module):
    """FFN con GELU activation (standard nei transformer moderni)."""

    def __init__(self, config: StudentConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.activation(self.fc1(x))))


class StudentTransformerLayer(nn.Module):
    """
    Un singolo layer Transformer con Pre-LayerNorm (più stabile del Post-LN).

    Pre-LN schema:
        x = x + Attention(LN(x))
        x = x + FFN(LN(x))
    """

    def __init__(self, config: StudentConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        if config.use_simplex_attention:
            self.attention = TwoSimplicialAttention(
                in_dim=config.hidden_size,
                out_dim=config.hidden_size,
                num_heads=config.num_attention_heads,
                dropout=config.dropout_prob,
                use_triton_kernel=getattr(config, "use_triton", True),
                with_residual=False, # Handled by StudentTransformerLayer
                with_norm=False,     # Handled by StudentTransformerLayer
                w1=config.w1,
                w2=config.w2
            )
        else:
            self.attention = StudentAttention(config)
        self.mlp = StudentFeedForward(config)
        self.resid_dropout = nn.Dropout(config.dropout_prob)
        # Use a buffer for is_bypassed to ensure synchronization across DDP ranks
        self.register_buffer("is_bypassed", torch.tensor(False, dtype=torch.bool))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Se il layer è bypassato, restituisce l'input invariato (residual identity)
        if self.is_bypassed:
            return hidden_states

        # Self-attention con residual
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        hidden_states = self.attention(
            hidden_states, 
            attention_mask=attention_mask,
        )
        hidden_states = residual + self.resid_dropout(hidden_states)

        # FFN con residual
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.resid_dropout(hidden_states)

        return hidden_states


# ---------------------------------------------------------------------------
# Modello principale
# ---------------------------------------------------------------------------


class StudentModel(PreTrainedModel):
    """
    Backbone del modello Student (senza LM head).
    Esposto separatamente per permettere feature-based KD sugli hidden states.
    """

    config_class = StudentConfig

    def __init__(self, config: StudentConfig):
        super().__init__(config)
        self.embeddings = StudentEmbeddings(config)
        self.layers = nn.ModuleList(
            [StudentTransformerLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.final_ln = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.gradient_checkpointing = False
        self.post_init()  # chiama _init_weights

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        hidden_states = self.embeddings(input_ids, attention_mask=attention_mask)

        all_hidden_states = []
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, 
                attention_mask=attention_mask,
            )
            all_hidden_states.append(hidden_states)

        hidden_states = self.final_ln(hidden_states)
        return hidden_states, tuple(all_hidden_states)


class StudentForCausalLM(PreTrainedModel, GenerationMixin):
    """
    Modello Student completo per Causal Language Modeling.

    Contratto HuggingFace:
    - Eredita PreTrainedModel (→ save/load, device management, .generate())
    - Restituisce CausalLMOutputWithPast (→ .logits accessibile dal KDTrainer)
    - lm_head ha out_features == config.vocab_size (critico per KD con teacher)
    """

    config_class = StudentConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: StudentConfig):
        super().__init__(config)
        # Backbone components directly at top level to match checkpoint
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.emb_dropout = nn.Dropout(config.dropout_prob)
        
        self.layers = nn.ModuleList(
            [StudentTransformerLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=1e-5)
        
        # LM head: hidden_size → vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()
        self.tie_weights()

    def get_input_embeddings(self):
        return self.token_embedding

    def set_input_embeddings(self, value):
        self.token_embedding = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def tie_weights(self, **kwargs):
        """Disabilitato per stabilità salvataggio in questa versione."""
        pass

    def _init_weights(self, module: nn.Module):
        """Inizializzazione standard HuggingFace (necessaria per post_init)."""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        token_type_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[CausalLMOutputWithPast, Tuple]:
        
        B, S = input_ids.shape
        device = input_ids.device

        if attention_mask is not None:
            position_ids = (torch.cumsum(attention_mask, dim=1) - 1).clamp(min=0)
        else:
            position_ids = torch.arange(S, dtype=torch.long, device=device).unsqueeze(0).expand(B, S)

        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(position_ids)
        hidden_states = self.emb_dropout(token_emb + pos_emb)
        
        all_hidden_states = () if output_hidden_states else None
        
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
            )
            
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            out = (logits,) + (all_hidden_states if output_hidden_states else ())
            return (loss,) + out if loss is not None else out

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=all_hidden_states,
            past_key_values=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        """Metodo richiesto da GenerationMixin per l'autoregressive decoding."""
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
