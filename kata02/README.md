# MLP Digits

## Task

Write a one hidden layer MLP, and a simple train loop for hand-written digits. Implement the following:

1. MLP, accepting input, hidden, output size. It should also provide a loss function.
2. NumpyDataset using torch dataset, accepting numpy inputs and targets
3. Train loop with AdamW optimizer, accepting `train_inputs, train_targets, valid_inputs, valid_targets, epochs`. It should return the valid predictions. Seeding will be handled by the test script.

Note: the dataset used in `run.ipynb` is datasets.load_digits which is actually the test dataset, but is used since it is available offline. It doesn't matter since this is for use as a toy dataset. It will be split into a train and valid set Accuracy on the valid set should be above 96% (and at most 99%).

If you wish to run on mnist, `run_mnist.ipynb` runs mnist, but requires internet and may take longer, and you may need to do more tuning. Accuracy should be above 94%.