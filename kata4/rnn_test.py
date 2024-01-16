import unittest
import torch
import torch.nn as nn
import kata4.rnn as sol

class RnnTestCase(unittest.TestCase):
    def check(self, rnn: nn.RNN, inputs: torch.Tensor, h_0: torch.Tensor):
        sol_rnn = sol.RNN(rnn.input_size, rnn.hidden_size)
        param_map = sol_rnn.get_param_mapping()
        state_dict = {param_map[k]: v for k, v in rnn.state_dict().items()}
        sol_rnn.load_state_dict(state_dict)

        outputs, h_n = rnn(inputs, h_0)
        expected = {}
        expected["outputs"] = outputs.detach().clone()
        expected["h_n"] = h_n.detach().clone()
        (outputs.sum() ** 2).backward()

        expected[f"inputs.grad"] = inputs.grad.detach().clone()
        expected[f"h_0.grad"] = h_0.grad.detach().clone()
        inputs.grad.zero_()
        h_0.grad.zero_()

        outputs, h_n = sol_rnn(inputs, h_0)
        (outputs.sum() ** 2).backward()

        torch.testing.assert_close(outputs, expected["outputs"])
        torch.testing.assert_close(h_n, expected["h_n"])
        torch.testing.assert_close(inputs.grad, expected["inputs.grad"])
        torch.testing.assert_close(h_0.grad, expected["h_0.grad"])
        sol_params = dict(sol_rnn.named_parameters())
        for k, v in rnn.named_parameters():
            torch.testing.assert_close(sol_params[param_map[k]].grad, v.grad)

    def test_single(self):
        torch.manual_seed(0)
        rnn = nn.RNN(10, 16, 2)
        inputs = torch.randn(1, 10, requires_grad=True)
        h_0 = torch.randn(2, 16, requires_grad=True)
        self.check(rnn, inputs, h_0)

    def test_unbatched(self):
        torch.manual_seed(1)
        rnn = nn.RNN(10, 16, 2)
        inputs = torch.randn(5, 10, requires_grad=True)
        h_0 = torch.randn(2, 16, requires_grad=True)
        self.check(rnn, inputs, h_0)

    def test_batched(self):
        torch.manual_seed(2)
        rnn = nn.RNN(10, 16, 2)
        inputs = torch.randn(5, 4, 10, requires_grad=True)
        h_0 = torch.randn(2, 4, 16, requires_grad=True)
        self.check(rnn, inputs, h_0)
