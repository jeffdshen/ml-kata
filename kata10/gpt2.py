from transformers import PreTrainedModel, GPT2Config

from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

import torch
import torch.nn as nn


class Gpt2PretrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPT2Config
    base_model_prefix = "gpt2"
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


class Gpt2LMHeadModel(Gpt2PretrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.config = config

        # TODO: Initialize model from config

        self.post_init()
        raise NotImplementedError("Gpt2Model not implemented")

    def get_input_embeddings(self):
        return self.transformer.wte

    def set_input_embeddings(self, value):
        self.transformer.wte = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def load_hf_state_dict(self, state_dict, *args, **kwargs):
        # TODO: Load HF state dict into our model. See below for reference.
        new_state_dict = {}
        non_layer_keys = [
            "transformer.wte.weight",
            "transformer.wpe.weight",
            "transformer.ln_f.weight",
            "transformer.ln_f.bias",
            "lm_head.weight",
        ]
        for k in non_layer_keys:
            new_state_dict[k] = state_dict[k]

        for i in range(self.config.num_hidden_layers):
            layer_keys = [
                f"transformer.h.{i}.ln_1.weight",
                f"transformer.h.{i}.ln_1.bias",
                f"transformer.h.{i}.attn.c_attn.weight",
                f"transformer.h.{i}.attn.c_attn.bias",
                f"transformer.h.{i}.attn.c_proj.weight",
                f"transformer.h.{i}.attn.c_proj.bias",
                f"transformer.h.{i}.ln_2.weight",
                f"transformer.h.{i}.ln_2.bias",
                f"transformer.h.{i}.mlp.c_fc.weight",
                f"transformer.h.{i}.mlp.c_fc.bias",
                f"transformer.h.{i}.mlp.c_proj.weight",
                f"transformer.h.{i}.mlp.c_proj.bias",
            ]

            new_state_dict.update({k: state_dict[k] for k in layer_keys})

        return self.load_state_dict(new_state_dict, *args, **kwargs)

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

        # TODO: Implement forward pass. See below for reference.
        # if labels is not None:
        #     labels = labels.to(logits.device)
        #     loss = torch.nn.functional.cross_entropy(
        #         logits[..., :-1, :].movedim(-1, 1), labels[..., 1:]
        #     )
        # else:
        #     loss = None
        # return CausalLMOutputWithCrossAttentions(
        #     loss=loss,  # type: ignore
        #     logits=logits,
        #     hidden_states=x.hidden_states,
        #     attentions=x.attentions,
        # )
        raise NotImplementedError("forward not implemented")


raise NotImplementedError("kata10/gpt2.py not implemented")
