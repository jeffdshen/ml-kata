# Llama Transformer

## Task

Implement a Llama transformer decoder model from scratch except attention.
You may refer to a multihead attention implementation (kata7, kata10, kata11). 

See llama.py for details.

Some notes:
1. Llama RoPE, instead of applying rotations to consecutive pairs, applies
rotations to corresponding pairs from the first and second half of the vector.
See `rotate_half` and `apply_rope` for hints.


The implementation will be compared against the llama huggingface model.

```
>>> transformers.AutoConfig.from_pretrained("meta-llama/Meta-Llama-3-8B")
LlamaConfig {
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 128000,
  "dtype": "bfloat16",
  "eos_token_id": 128001,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "max_position_embeddings": 8192,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 500000.0,
  "tie_word_embeddings": false,
  "transformers_version": "4.57.0",
  "use_cache": true,
  "vocab_size": 128256
}
>>> transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", dtype="auto", device_map="auto")
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 4096)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((4096,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=4096, out_features=128256, bias=False)
)
>>> list(dict.fromkeys([re.sub("\\.\\d+\\.", ".{i}.", key) for key in model.state_dict()]))
['model.embed_tokens.weight',
 'model.layers.{i}.self_attn.q_proj.weight',
 'model.layers.{i}.self_attn.k_proj.weight',
 'model.layers.{i}.self_attn.v_proj.weight',
 'model.layers.{i}.self_attn.o_proj.weight',
 'model.layers.{i}.mlp.gate_proj.weight',
 'model.layers.{i}.mlp.up_proj.weight',
 'model.layers.{i}.mlp.down_proj.weight',
 'model.layers.{i}.input_layernorm.weight',
 'model.layers.{i}.post_attention_layernorm.weight',
 'model.norm.weight',
 'lm_head.weight']
```