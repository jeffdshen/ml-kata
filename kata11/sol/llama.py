from transformers import PreTrainedModel, LlamaConfig

from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
)

import torch
import torch.nn as nn


class LlamaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LlamaConfig
    base_model_prefix = "llama"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        pass


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        exponents = torch.arange(0, config.head_dim, 2) / config.head_dim
        inv_freq = 1 / (config.rope_theta ** (exponents))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor):
        position_ids = position_ids.to(dtype=self.inv_freq.dtype).unsqueeze(-1)
        x = position_ids @ self.inv_freq.unsqueeze(-2)
        x = torch.cat([x, x], dim=-1)
        cos = x.cos()
        sin = x.sin()
        return cos, sin


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.q_dim = config.head_dim * config.num_attention_heads
        self.kv_dim = config.head_dim * config.num_key_value_heads
        self.group_size = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(
            config.hidden_size, self.q_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.kv_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.kv_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.q_dim, config.hidden_size, bias=config.attention_bias
        )
        self.rope_theta = config.rope_theta

    def to_multi_head(self, x):
        x = x.reshape(x.size()[:-1] + (-1, self.head_dim))
        x = x.movedim(-2, 0)
        return x

    def from_multi_head(self, x):
        x = x.movedim(0, -2)
        x = x.reshape(x.size()[:-2] + (-1,))
        return x

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        x = hidden_state
        q = self.to_multi_head(self.q_proj(x))
        k = self.to_multi_head(self.k_proj(x))
        v = self.to_multi_head(self.v_proj(x))

        q, k = apply_rope(q, k, cos, sin)
        k = k.repeat_interleave(self.group_size, dim=0)
        v = v.repeat_interleave(self.group_size, dim=0)

        x = q @ k.transpose(-1, -2)
        x /= q.size(-1) ** 0.5
        seq_len = attention_mask.size(-1)
        attention_mask = attention_mask.unsqueeze(-2).expand(-1, seq_len, seq_len)
        attention_mask = 1 - torch.tril(attention_mask)
        attention_mask.masked_fill_(attention_mask != 0.0, float("-inf"))
        x += attention_mask
        x = torch.nn.functional.softmax(x, dim=-1)
        attention = x
        x = x @ v
        x = self.from_multi_head(x)
        x = self.o_proj(x)

        return x, attention.transpose(0, 1)


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.mlp_bias
        )
        self.silu = nn.SiLU()

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        x = hidden_state
        x = self.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return x


class LlamaRmsNorm(nn.Module):
    def __init__(self, normalized_shape: tuple[int], eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps
        self.dims = [-(i + 1) for i in range(len(normalized_shape))]

    def forward(self, x: torch.Tensor):
        inv_rms = torch.rsqrt((x**2).mean(dim=self.dims, keepdim=True) + self.eps)
        x = x * inv_rms
        x = x * self.weight
        return x


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRmsNorm(
            (config.hidden_size,), eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = LlamaRmsNorm(
            (config.hidden_size,), eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        x = hidden_state
        a, b = self.self_attn(self.input_layernorm(x), attention_mask, cos, sin)
        x = a + x
        x = self.mlp(self.post_attention_layernorm(x), attention_mask) + x
        return x, b


class LlamaModel(PreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRmsNorm((config.hidden_size,), eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None,
    ) -> BaseModelOutputWithPast:
        if position_ids is None:
            position_ids = torch.arange(input_ids.size(-1), device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        x = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(position_ids)
        hidden_states: list[torch.FloatTensor] = []
        attentions: list[torch.FloatTensor] = []
        for layer in self.layers:
            hidden_states.append(x)
            x, attention = layer(x, attention_mask, cos, sin)
            attentions.append(attention)

        x = self.norm(x)
        hidden_states.append(x)
        return BaseModelOutputWithPast(
            last_hidden_state=x,
            hidden_states=tuple(hidden_states),
            attentions=tuple(attentions),
        )


class LlamaForCausalLM(PreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(
            in_features=config.hidden_size, out_features=config.vocab_size, bias=False
        )

        self.post_init()

    def load_hf_state_dict(self, state_dict, *args, **kwargs):
        # TODO: Load HF state dict into our model. See below for reference.
        return self.load_state_dict(state_dict, *args, **kwargs, strict=False)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> CausalLMOutputWithPast:
        if input_ids is None:
            raise ValueError("input_ids is required for this model")

        if attention_mask is None:
            raise ValueError("attention_mask is required for this model")

        x: BaseModelOutputWithPast = self.model(input_ids, attention_mask, position_ids)
        logits = self.lm_head(x.last_hidden_state)
        assert logits is not None
        # TODO: Implement forward pass. See below for reference.
        if labels is not None:
            labels = labels.to(logits.device)
            loss = torch.nn.functional.cross_entropy(
                logits[..., :-1, :].movedim(-1, 1), labels[..., 1:]
            )
        else:
            loss = None
        return CausalLMOutputWithPast(
            loss=loss,  # type: ignore
            logits=logits,
            hidden_states=x.hidden_states,
            attentions=x.attentions,
        )
