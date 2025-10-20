import unittest
import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM as hf_LlamaForCausalLM,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    import kata11.llama as sol
except NotImplementedError:
    import kata11.sol.llama as sol


class LlamaTestCase(unittest.TestCase):
    def check_forward(
        self,
        stage: str,
        hf_model: hf_LlamaForCausalLM,
        sol_model: sol.LlamaForCausalLM,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ):
        with torch.no_grad():
            hf_output: CausalLMOutputWithPast = hf_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                output_attentions=True,
            )
            sol_output: CausalLMOutputWithPast = sol_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                output_attentions=True,
            )
        assert hf_output.hidden_states is not None
        assert hf_output.attentions is not None
        for i in range(len(hf_output.hidden_states)):
            if sol_output.hidden_states is not None:
                self.assertEqual(
                    len(hf_output.hidden_states), len(sol_output.hidden_states)
                )
                actual = sol_output.hidden_states[i]
                expected = hf_output.hidden_states[i]
                torch.testing.assert_close(
                    actual,
                    expected,
                    msg=lambda msg: f"{stage}: Hidden state {i} mismatch: {msg}, actual={actual}, expected={expected}",
                )
            if sol_output.attentions is not None:
                self.assertEqual(len(hf_output.attentions), len(sol_output.attentions))
                if i < len(hf_output.attentions):
                    actual = sol_output.attentions[i]
                    expected = hf_output.attentions[i]
                    torch.testing.assert_close(
                        actual,
                        expected,
                        msg=lambda msg: f"{stage}: Attention {i} mismatch: {msg}, actual={actual}, expected={expected}",
                    )

        torch.testing.assert_close(sol_output.logits, hf_output.logits)
        torch.testing.assert_close(sol_output.loss, hf_output.loss)

    def check(
        self,
        config: LlamaConfig,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ):
        """
        Compare the solution RoBERTa implementation with HuggingFace RoBERTa (random initialization)
        """
        # Load both models with same weights and check that first forward pass matches
        hf_model = hf_LlamaForCausalLM(config)
        sol_model = sol.LlamaForCausalLM(config)
        sol_model.load_hf_state_dict(hf_model.state_dict())

        hf_model.eval()
        sol_model.eval()

        self.check_forward("first forward", hf_model, sol_model, input_ids, attention_mask, position_ids)
        # Run a backward pass, and check that the second forward pass matches
        lr = 0.1
        hf_optimizer = torch.optim.SGD(hf_model.parameters(), lr=lr)
        sol_optimizer = torch.optim.SGD(sol_model.parameters(), lr=lr)

        hf_output: CausalLMOutputWithPast = hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=input_ids,
        )
        sol_output: CausalLMOutputWithPast = sol_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=input_ids,
        )

        assert hf_output.loss is not None
        assert sol_output.loss is not None
        hf_output.loss.backward()
        sol_output.loss.backward()
        sol_params = dict(sol_model.named_parameters())
        for name, param in hf_model.named_parameters():
            if not param.requires_grad:
                continue
            if name in sol_params:
                actual = sol_params[name].grad
                expected = param.grad
                torch.testing.assert_close(
                    actual,
                    expected,
                    msg=lambda msg: f"Grad {name} mismatch: {msg}, actual={actual}, expected={expected}",
                )
        hf_optimizer.step()
        sol_optimizer.step()
        hf_optimizer.zero_grad()
        sol_optimizer.zero_grad()

        self.check_forward("second forward", hf_model, sol_model, input_ids, attention_mask)

    def get_config(self):
        return LlamaConfig(
            vocab_size=100,
            max_position_embeddings=128,
            hidden_size=64,
            intermediate_size=224,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            hidden_act="silu",
            attention_dropout=0.0,
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
            use_cache=True,
            tie_word_embeddings=False,
            attn_implementation="eager",
        )

    def test_basic(self):
        """Test with fixed position ids."""
        torch.manual_seed(42)

        config = self.get_config()
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(3, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        position_ids = (
            (torch.arange(seq_len) + 2).unsqueeze(0).expand_as(attention_mask)
        )
        self.check(config, input_ids, attention_mask, position_ids)

    def test_none_position_ids(self):
        """Test with a small configuration for faster execution."""
        torch.manual_seed(42)

        config = self.get_config()
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(3, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        self.check(config, input_ids, attention_mask)

    def test_single_sequence(self):
        """Test with a single sequence."""
        torch.manual_seed(0)

        config = self.get_config()
        input_ids = torch.randint(3, config.vocab_size, (1, 62))
        attention_mask = torch.ones(1, 62)

        self.check(config, input_ids, attention_mask)

    def test_with_padding(self):
        """Test with attention mask that includes padding."""
        torch.manual_seed(1)

        config = self.get_config()
        batch_size, seq_len = 3, 24
        input_ids = torch.randint(3, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[0, 20:] = 0
        attention_mask[1, 18:] = 0

        self.check(config, input_ids, attention_mask)
