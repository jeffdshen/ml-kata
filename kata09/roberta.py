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


class RobertaModel(RobertaPretrainedModel):
    def __init__(self, config: RobertaConfig):
        super().__init__(config)
        self.config = config

        # TODO: Initialize model from config, config args given for reference.
        self.vocab_size = self.config.vocab_size or 50265
        self.hidden_size = self.config.hidden_size or 768
        self.num_hidden_layers = self.config.num_hidden_layers or 12
        self.num_attention_heads = self.config.num_attention_heads or 12
        self.intermediate_size = self.config.intermediate_size or 3072
        self.hidden_act = self.config.hidden_act or "gelu"
        self.hidden_dropout_prob = self.config.hidden_dropout_prob or 0.1
        self.attention_probs_dropout_prob = (
            self.config.attention_probs_dropout_prob or 0.1
        )
        self.max_position_embeddings = self.config.max_position_embeddings or 512
        self.initializer_range = self.config.initializer_range or 0.02
        self.layer_norm_eps = self.config.layer_norm_eps or 1e-12
        self.pad_token_id = self.config.pad_token_id or 1
        self.bos_token_id = self.config.bos_token_id or 0
        self.eos_token_id = self.config.eos_token_id or 2
        self.position_embedding_type = self.config.position_embedding_type or "absolute"

        raise NotImplementedError("RobertaModel not implemented")

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

        # return self.load_state_dict(new_state_dict, *args, **kwargs)
        raise NotImplementedError("load_hf_state_dict not implemented")

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

        # TODO: Implement forward pass. See below for reference.
        # return BaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=x, hidden_states=y)
        raise NotImplementedError("forward not implemented")
