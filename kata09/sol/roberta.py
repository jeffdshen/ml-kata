import torch
from transformers import PreTrainedModel, RobertaConfig
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


class RobertaPretrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RobertaConfig
    base_model_prefix = "roberta"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        pass


class RobertaEmbeddings(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int,
        type_vocab_size: int,
        pad_token_id: int,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings

        self.word_embeddings = torch.nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_token_id
        )
        self.position_embeddings = torch.nn.Embedding(
            max_position_embeddings, hidden_size
        )
        self.token_type_embeddings = torch.nn.Embedding(type_vocab_size, hidden_size)
        self.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor):
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(torch.zeros_like(input_ids))
        embeddings = word_embeddings + token_type_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        return embeddings


class RobertSelfAttention(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float,
    ):
        super().__init__()
        self.query = torch.nn.Linear(hidden_size, hidden_size)
        self.key = torch.nn.Linear(hidden_size, hidden_size)
        self.value = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(attention_probs_dropout_prob)
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_attention_heads

    def to_multi_head(self, x: torch.Tensor):
        x = x.reshape(*x.size()[:-1], self.num_attention_heads, self.head_size)
        x = x.unsqueeze(0).transpose(0, -2).squeeze(-2)
        return x

    def from_multi_head(self, x: torch.Tensor):
        x = x.unsqueeze(-2).transpose(0, -2).squeeze(0)
        x = x.reshape(*x.size()[:-2], self.hidden_size)
        return x

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        q = self.to_multi_head(self.query(hidden_states))
        k = self.to_multi_head(self.key(hidden_states))
        v = self.to_multi_head(self.value(hidden_states))

        x: torch.Tensor = q @ k.transpose(-1, -2)
        x /= self.head_size**0.5
        attention_mask = 1 - attention_mask.unsqueeze(-2)
        attention_mask.masked_fill_(attention_mask != 0.0, float("-inf"))
        x += attention_mask
        x = torch.nn.functional.softmax(x, dim=-1)
        x = self.dropout(x)
        x = x @ v

        x = self.from_multi_head(x)
        return x


class RobertaSelfOutput(torch.nn.Module):
    def __init__(
        self, hidden_size: int, hidden_dropout_prob: float, layer_norm_eps: float
    ):
        super().__init__()
        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        x = self.dense(hidden_states)
        x = self.dropout(x)
        x = self.LayerNorm(x + input_tensor)
        return x


class RobertaAttention(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float,
        hidden_dropout_prob: float,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.self = RobertSelfAttention(
            hidden_size, num_attention_heads, attention_probs_dropout_prob
        )
        self.output = RobertaSelfOutput(
            hidden_size, hidden_dropout_prob, layer_norm_eps
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        x = self.self(hidden_states, attention_mask)
        x = self.output(x, hidden_states)
        return x


class RobertaIntermediate(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.dense = torch.nn.Linear(hidden_size, intermediate_size)
        self.act_fn = torch.nn.GELU() if hidden_act == "gelu" else torch.nn.ReLU()

    def forward(self, hidden_states: torch.Tensor):
        x = self.dense(hidden_states)
        x = self.act_fn(x)
        return x


class RobertaOutput(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_dropout_prob: float,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.dense = torch.nn.Linear(intermediate_size, hidden_size)
        self.dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        x = self.dense(hidden_states)
        x = self.dropout(x)
        x = self.LayerNorm(x + input_tensor)
        return x


class RobertaLayer(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float,
        intermediate_size: int,
        hidden_act: str,
        hidden_dropout_prob: float,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.act_fn = torch.nn.GELU() if hidden_act == "gelu" else torch.nn.ReLU()
        self.intermediate = RobertaIntermediate(
            hidden_size, intermediate_size, hidden_act
        )
        self.attention = RobertaAttention(
            hidden_size,
            num_attention_heads,
            attention_probs_dropout_prob,
            hidden_dropout_prob,
            layer_norm_eps,
        )
        self.output = RobertaOutput(
            hidden_size, intermediate_size, hidden_dropout_prob, layer_norm_eps
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        x = self.attention(hidden_states, attention_mask)
        x = self.output(self.intermediate(x), x)
        return x


class RobertaEncoder(torch.nn.Module):
    def __init__(
        self,
        num_hidden_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float,
        intermediate_size: int,
        hidden_act: str,
        hidden_dropout_prob: float,
        layer_norm_eps: float,
    ):
        super().__init__()

        self.layer = torch.nn.ModuleList(
            [
                RobertaLayer(
                    hidden_size,
                    num_attention_heads,
                    attention_probs_dropout_prob,
                    intermediate_size,
                    hidden_act,
                    hidden_dropout_prob,
                    layer_norm_eps,
                )
                for _ in range(num_hidden_layers)
            ]
        )

    def forward(
        self,
        hidden_state: torch.FloatTensor,
        attention_mask: torch.Tensor,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutputWithPoolingAndCrossAttentions:
        hidden_states: list[torch.FloatTensor] = []
        if output_hidden_states:
            hidden_states.append(hidden_state)
        for layer in self.layer:
            hidden_state = layer(hidden_state, attention_mask)
            if output_hidden_states:
                hidden_states.append(hidden_state)

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_state,
            hidden_states=tuple(hidden_states) if output_hidden_states else None,  # type: ignore
        )


class RobertaModel(RobertaPretrainedModel):
    def __init__(self, config: RobertaConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddings(
            config.vocab_size,
            config.hidden_size,
            config.max_position_embeddings,
            config.type_vocab_size,
            config.pad_token_id,
            config.layer_norm_eps,
        )
        self.encoder = RobertaEncoder(
            config.num_hidden_layers,
            config.hidden_size,
            config.num_attention_heads,
            config.attention_probs_dropout_prob,
            config.intermediate_size,
            config.hidden_act,
            config.hidden_dropout_prob,
            config.layer_norm_eps,
        )

    def load_hf_state_dict(self, state_dict, *args, **kwargs):
        # TODO: Load HF state dict into our model. See below for reference.
        new_state_dict = {}
        non_layer_keys = [
            "embeddings.LayerNorm.bias",
            "embeddings.LayerNorm.weight",
            "embeddings.position_embeddings.weight",
            "embeddings.token_type_embeddings.weight",
            "embeddings.word_embeddings.weight",
            # "pooler.dense.bias",
            # "pooler.dense.weight",
        ]
        for k in non_layer_keys:
            new_state_dict[k] = state_dict[k]

        for i in range(self.config.num_hidden_layers):
            layer_keys = [
                f"encoder.layer.{i}.attention.output.LayerNorm.bias",
                f"encoder.layer.{i}.attention.output.LayerNorm.weight",
                f"encoder.layer.{i}.attention.output.dense.bias",
                f"encoder.layer.{i}.attention.output.dense.weight",
                f"encoder.layer.{i}.attention.self.key.bias",
                f"encoder.layer.{i}.attention.self.key.weight",
                f"encoder.layer.{i}.attention.self.query.bias",
                f"encoder.layer.{i}.attention.self.query.weight",
                f"encoder.layer.{i}.attention.self.value.bias",
                f"encoder.layer.{i}.attention.self.value.weight",
                f"encoder.layer.{i}.intermediate.dense.bias",
                f"encoder.layer.{i}.intermediate.dense.weight",
                f"encoder.layer.{i}.output.LayerNorm.bias",
                f"encoder.layer.{i}.output.LayerNorm.weight",
                f"encoder.layer.{i}.output.dense.bias",
                f"encoder.layer.{i}.output.dense.weight",
            ]

            new_state_dict.update({k: state_dict[k] for k in layer_keys})

        return self.load_state_dict(new_state_dict, *args, **kwargs)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutputWithPoolingAndCrossAttentions:
        if input_ids is None:
            raise ValueError("input_ids is required for this model")

        if position_ids is None:
            position_ids = (
                torch.arange(
                    input_ids.size(-1),
                    device=input_ids.device,
                )
                + self.config.pad_token_id
                + 1
            )
        x = self.embeddings(input_ids, position_ids)
        x = self.encoder(x, attention_mask, output_attentions, output_hidden_states)
        return x
