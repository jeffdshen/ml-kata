import unittest
import torch

from kata13.sol.deepseek_moe import MoE, ModelArgs

try:
    import kata13.moe as sol
except NotImplementedError:
    import kata13.sol.moe as sol


class MoeTestCase(unittest.TestCase):
    def check_forward(
        self,
        ds_moe: MoE,
        sol_moe: sol.MoE,
        x: torch.Tensor,
    ):
        with torch.no_grad():
            ds_y = ds_moe(x)
            sol_y = sol_moe(x)
        torch.testing.assert_close(ds_y, sol_y)

    def check(
        self,
        config: ModelArgs,
        x: torch.Tensor,
    ):
        """
        Compare the solution RoBERTa implementation with HuggingFace RoBERTa (random initialization)
        """
        # Load both models with same weights and check that first forward pass matches
        hf_model = MoE(config)
        sol_model = sol.MoE(config)
        sol_model.load_ds_state_dict(hf_model.state_dict())

        hf_model.eval()
        sol_model.eval()

        self.check_forward(hf_model, sol_model, x)

    def get_config(self):
        return ModelArgs(
            vocab_size=100,
            dim=32,
            inter_dim=128,
            moe_inter_dim=22,
            n_routed_experts=8,
            n_shared_experts=2,
            n_activated_experts=3,
            n_expert_groups=1,
            n_limited_groups=1,
            score_func="sigmoid",
            route_scale=1.0,
        )

    def test_basic(self):
        """Test with fixed position ids."""
        torch.manual_seed(42)

        config = self.get_config()
        batch_size, seq_len = 2, 32
        x = torch.randn(batch_size, seq_len, config.dim, dtype=torch.bfloat16) / 100
        self.check(config, x)

    def test_single_sequence(self):
        """Test with a single sequence."""
        torch.manual_seed(0)

        config = self.get_config()
        x = torch.randn(1, 62, config.dim, dtype=torch.bfloat16)
        self.check(config, x)

    def test_no_batch(self):
        """Test with no batch dimension"""
        torch.manual_seed(0)

        config = self.get_config()
        x = torch.randn(62, config.dim, dtype=torch.bfloat16)
        self.check(config, x)
