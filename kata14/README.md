# MLP Digits (No Autograd)

## Task

This is a copy of kata02, but the goal is to write a whole train loop from scratch (no autograd, basic ops only). The method should return the predictions for the valid inputs, seeding will be handled by the test script. The following are suggestions, but you can use any method that hits the required accuracy in `train_test.py`:

1. Use a one hidden layer MLP, hidden size 256.
2. Use SGD with typical parameters.
3. You may use torch or numpy (try both!), but no autograd is allowed.
4. Better initialization may help.
5. Use batch size 16, learning rate 0.01.

There will be a few test cases including XOR, mnist, mnist (where a dummy valid targets is passed to prevent leakage). `run_mnist.ipynb` is also provided again, but requires the internet, and may take longer.