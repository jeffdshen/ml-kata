# GPT2 Transformer

## Task

Implement a GPT2 transformer decoder model from scratch except attention.
You may refer to a multihead attention implementation (kata7, kata10). 

See gpt2.py for details.

Some notes:
1. (Given) Weights are tied using `self.post_init` and set/get embeddings methods.
2. (Given) `Conv1D` is the same as a `Linear`, except the shape is transposed.
3. (Given) For the loss, the labels are the same shape, but shifted.
Namely, the first n-1 logits should match the last n-1 labels.
Weirdly, labels are ignored via -100 in inputs, and are not set via attention mask.
NOTE: that the classes are in dim 1, not -1!
4. `NewGELUActivation` is in `torch` as `GELU(approximate="tanh")`.
5. `n_inner` defaults to `4 * n_embd`.
6. `c_attn` has `3 * n_embd` out features, and `n_embd` in features.

The implementation will be compared against the gpt2 huggingface model.

```
>>> transformers.AutoConfig.from_pretrained("gpt2")
GPT2Config {
  "_name_or_path": "gpt2",
  "activation_function": "gelu_new",
  "architectures": [
    "GPT2LMHeadModel"
  ],
  "attn_pdrop": 0.1,
  "bos_token_id": 50256,
  "embd_pdrop": 0.1,
  "eos_token_id": 50256,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "model_type": "gpt2",
  "n_ctx": 1024,
  "n_embd": 768,
  "n_head": 12,
  "n_inner": null,
  "n_layer": 12,
  "n_positions": 1024,
  "reorder_and_upcast_attn": false,
  "resid_pdrop": 0.1,
  "scale_attn_by_inverse_layer_idx": false,
  "scale_attn_weights": true,
  "summary_activation": null,
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": true,
  "summary_type": "cls_index",
  "summary_use_proj": true,
  "task_specific_params": {
    "text-generation": {
      "do_sample": true,
      "max_length": 50
    }
  },
  "transformers_version": "4.32.1",
  "use_cache": true,
  "vocab_size": 50257
}
>>> transformers.AutoModelForCausalLM.from_pretrained("gpt2")
GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(50257, 768)
    (wpe): Embedding(1024, 768)
    (drop): Dropout(p=0.1, inplace=False)
    (h): ModuleList(
      (0-11): 12 x GPT2Block(
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attn): GPT2Attention(
          (c_attn): Conv1D()
          (c_proj): Conv1D()
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (resid_dropout): Dropout(p=0.1, inplace=False)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): GPT2MLP(
          (c_fc): Conv1D()
          (c_proj): Conv1D()
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=768, out_features=50257, bias=False)
)
>>> list(dict.fromkeys([re.sub("\\.\\d+\\.", ".{i}.", key) for key in model.state_dict()]))
['transformer.wte.weight',
 'transformer.wpe.weight',
 'transformer.h.{i}.ln_1.weight',
 'transformer.h.{i}.ln_1.bias',
 'transformer.h.{i}.attn.c_attn.weight',
 'transformer.h.{i}.attn.c_attn.bias',
 'transformer.h.{i}.attn.c_proj.weight',
 'transformer.h.{i}.attn.c_proj.bias',
 'transformer.h.{i}.ln_2.weight',
 'transformer.h.{i}.ln_2.bias',
 'transformer.h.{i}.mlp.c_fc.weight',
 'transformer.h.{i}.mlp.c_fc.bias',
 'transformer.h.{i}.mlp.c_proj.weight',
 'transformer.h.{i}.mlp.c_proj.bias',
 'transformer.ln_f.weight',
 'transformer.ln_f.bias',
 'lm_head.weight']
```