import unittest
import torch
from transformers import RobertaConfig, RobertaModel as hf_RobertaModel

try:
    import kata9.roberta as sol
except NotImplementedError:
    import kata9.sol.roberta as sol


class RobertaTestCase(unittest.TestCase):
    def check_forward(
        self,
        hf_model: hf_RobertaModel,
        sol_model: sol.RobertaModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ):
        with torch.no_grad():
            hf_output = hf_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
            )
            sol_output = sol_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
            )
        self.assertEqual(len(hf_output.hidden_states), len(sol_output.hidden_states))
        for i in range(len(hf_output.hidden_states)):
            actual = sol_output.hidden_states[i]
            expected = hf_output.hidden_states[i]
            torch.testing.assert_close(
                actual,
                expected,
                msg=lambda msg: f"Hidden state {i} mismatch: {msg}, actual={actual}, expected={expected}",
            )

        torch.testing.assert_close(
            sol_output.last_hidden_state, hf_output.last_hidden_state
        )

    def check(
        self,
        config: RobertaConfig,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ):
        """
        Compare the solution RoBERTa implementation with HuggingFace RoBERTa (random initialization)
        """
        # Load both models with same weights and check that first forward pass matches
        hf_model = hf_RobertaModel(config)
        sol_model = sol.RobertaModel(config)
        sol_model.load_hf_state_dict(hf_model.state_dict())

        hf_model.eval()
        sol_model.eval()

        self.check_forward(hf_model, sol_model, input_ids, attention_mask, position_ids)
        # Run a backward pass, and check that the second forward pass matches
        lr = 0.1
        hf_optimizer = torch.optim.SGD(hf_model.parameters(), lr=lr)
        sol_optimizer = torch.optim.SGD(sol_model.parameters(), lr=lr)

        hf_output_train = hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        sol_output_train = sol_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        hf_loss = hf_output_train.last_hidden_state.sum()
        sol_loss = sol_output_train.last_hidden_state.sum()
        hf_loss.backward()
        sol_loss.backward()
        hf_optimizer.step()
        sol_optimizer.step()
        hf_optimizer.zero_grad()
        sol_optimizer.zero_grad()

        self.check_forward(hf_model, sol_model, input_ids, attention_mask)

    def get_config(self):
        return RobertaConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            max_position_embeddings=128,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
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
