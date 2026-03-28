import torch

class ReluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        ctx.save_for_backward(inputs)
        return (inputs > 0) * inputs

    @staticmethod
    def backward(ctx, d_outputs):
        inputs, = ctx.saved_tensors
        return (inputs > 0) * d_outputs