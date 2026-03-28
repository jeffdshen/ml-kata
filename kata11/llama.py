import torch
from transformers import LlamaConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


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
    # where cos and sin are repeated matrices
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class LlamaForCausalLM(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.config = config

        # TODO: Initialize model from config

        self.post_init()
        raise NotImplementedError("__init__ not implemented")

    def load_hf_state_dict(self, state_dict, *args, **kwargs):
        # TODO: Load HF state dict into our model. See below for reference.
        return self.load_state_dict(state_dict, *args, **kwargs)

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

        # TODO: Implement forward pass. See below for reference.
        # if labels is not None:
        #     labels = labels.to(logits.device)
        #     loss = torch.nn.functional.cross_entropy(
        #         logits[..., :-1, :].movedim(-1, 1), labels[..., 1:]
        #     )
        # else:
        #     loss = None
        # return CausalLMOutputWithPast(
        #     loss=loss,  # type: ignore
        #     logits=logits,
        #     hidden_states=x.hidden_states,
        #     attentions=x.attentions,
        # )
        raise NotImplementedError("forward not implemented")


raise NotImplementedError("kata11/llama.py not implemented")
