import torch
import torch.nn as nn
from transformers import GPT2Config, PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)


class Gpt2PretrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPT2Config
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        pass


class Conv1D(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((in_features, out_features)))
        self.bias = nn.Parameter(torch.empty(out_features))

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight.transpose(-1, -2), self.bias)


class GPT2Attention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_attn = Conv1D(config.n_embd, 3 * config.n_embd)
        self.c_proj = Conv1D(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.n_heads = config.n_head

    def to_multi_head(self, x):
        x = x.reshape(x.size()[:-1] + (self.n_heads, -1))
        x = x.movedim(-2, 0)
        return x

    def from_multi_head(self, x):
        x = x.movedim(0, -2)
        x = x.reshape(x.size()[:-2] + (-1,))
        return x

    def forward(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor):
        x = hidden_state
        q, k, v = self.c_attn(x).split(x.size(-1), dim=-1)
        q, k, v = self.to_multi_head(q), self.to_multi_head(k), self.to_multi_head(v)

        x = q @ k.transpose(-1, -2)
        x /= q.size(-1) ** 0.5
        seq_len = attention_mask.size(-1)
        attention_mask = attention_mask.unsqueeze(-2).expand(-1, seq_len, seq_len)
        attention_mask = 1 - torch.tril(attention_mask)
        attention_mask.masked_fill_(attention_mask != 0.0, float("-inf"))
        x += attention_mask
        x = torch.nn.functional.softmax(x, dim=-1)
        x = self.attn_dropout(x)
        attention = x
        x = x @ v
        x = self.from_multi_head(x)
        x = self.c_proj(x)
        x = self.resid_dropout(x)
        return x, attention.transpose(0, 1)


class GPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        n_inner = config.n_inner or 4 * config.n_embd
        self.c_fc = Conv1D(config.n_embd, n_inner)
        self.c_proj = Conv1D(n_inner, config.n_embd)

        if config.activation_function == "gelu_new":
            self.act = nn.GELU(approximate="tanh")
        else:
            raise ValueError(
                f"Unknown activation function {config.activation_function}"
            )
        self.dropout = nn.Dropout(config.embd_pdrop)

    def forward(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor):
        x = hidden_state
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Gpt2Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)

    def forward(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor):
        x = hidden_state
        a, b = self.attn(self.ln_1(x), attention_mask)
        x = a + x
        x = self.mlp(self.ln_2(x), attention_mask) + x
        return x, b


class Gpt2Model(Gpt2PretrainedModel):
    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(Gpt2Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd, config.layer_norm_epsilon)

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, value):
        self.wte = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        if position_ids is None:
            position_ids = torch.arange(input_ids.size(-1), device=input_ids.device)

        hidden_states = []
        attentions = []
        x = self.wte(input_ids) + self.wpe(position_ids)
        x = self.drop(x)
        for layer in self.h:
            hidden_states.append(x)
            x, attn = layer(x, attention_mask)
            attentions.append(attn)

        x = self.ln_f(x)
        hidden_states.append(x)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=x,
            hidden_states=hidden_states,  # type: ignore
            attentions=attentions,  # type: ignore
        )


class Gpt2LMHeadModel(Gpt2PretrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: GPT2Config):
        super().__init__(config)

        self.transformer = Gpt2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.config = config
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def load_hf_state_dict(self, state_dict, *args, **kwargs):
        return self.load_state_dict(state_dict, *args, **kwargs)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> CausalLMOutputWithCrossAttentions:
        if input_ids is None:
            raise ValueError("input_ids is required for this model")

        x: BaseModelOutputWithPastAndCrossAttentions = self.transformer(
            input_ids,
            attention_mask,
            position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        logits = self.lm_head(x.last_hidden_state)
        if labels is not None:
            labels = labels.to(logits.device)
            loss = torch.nn.functional.cross_entropy(
                logits[..., :-1, :].movedim(-1, 1), labels[..., 1:]
            )
        else:
            loss = None
        return CausalLMOutputWithCrossAttentions(
            loss=loss,  # type: ignore
            logits=logits,
            hidden_states=x.hidden_states,
            attentions=x.attentions,
        )
