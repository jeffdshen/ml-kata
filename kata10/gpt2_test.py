import unittest
import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel as hf_GPT2LMHeadModel,
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

try:
    import kata10.gpt2 as sol
except NotImplementedError:
    import kata10.sol.gpt2 as sol


class Gpt2TestCase(unittest.TestCase):
    def check_forward(
        self,
        hf_model: hf_GPT2LMHeadModel,
        sol_model: sol.Gpt2LMHeadModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ):
        with torch.no_grad():
            hf_output: CausalLMOutputWithCrossAttentions = hf_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                output_attentions=True,
            )
            sol_output: CausalLMOutputWithCrossAttentions = sol_model(
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
                    msg=lambda msg: f"Hidden state {i} mismatch: {msg}, actual={actual}, expected={expected}",
                )
            if sol_output.attentions is not None:
                self.assertEqual(len(hf_output.attentions), len(sol_output.attentions))
                if i < len(hf_output.attentions):
                    actual = sol_output.attentions[i]
                    expected = hf_output.attentions[i]
                    torch.testing.assert_close(
                        actual,
                        expected,
                        msg=lambda msg: f"Attention {i} mismatch: {msg}, actual={actual}, expected={expected}",
                    )

        torch.testing.assert_close(sol_output.logits, hf_output.logits)
        torch.testing.assert_close(sol_output.loss, hf_output.loss)

    def check(
        self,
        config: GPT2Config,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ):
        """
        Compare the solution RoBERTa implementation with HuggingFace RoBERTa (random initialization)
        """
        # Load both models with same weights and check that first forward pass matches
        hf_model = hf_GPT2LMHeadModel(config)
        sol_model = sol.Gpt2LMHeadModel(config)
        sol_model.load_hf_state_dict(hf_model.state_dict())

        hf_model.eval()
        sol_model.eval()

        self.check_forward(hf_model, sol_model, input_ids, attention_mask, position_ids)
        # Run a backward pass, and check that the second forward pass matches
        lr = 0.1
        hf_optimizer = torch.optim.SGD(hf_model.parameters(), lr=lr)
        sol_optimizer = torch.optim.SGD(sol_model.parameters(), lr=lr)

        hf_output: CausalLMOutputWithCrossAttentions = hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=input_ids,
        )
        sol_output: CausalLMOutputWithCrossAttentions = sol_model(
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

        self.check_forward(hf_model, sol_model, input_ids, attention_mask)

    def get_config(self):
        return GPT2Config(
            vocab_size=100,
            n_positions=128,
            n_embd=64,
            n_layer=2,
            n_head=4,
            n_inner=None,
            activation_function="gelu_new",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,
            scale_attn_weights=True,
            use_cache=False,
        )

    def test_basic(self):
        """Test with fixed position ids."""
        torch.manual_seed(42)

        config = self.get_config()
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(3, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        position_ids = torch.arange(seq_len) + 2

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
