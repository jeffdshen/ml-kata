# ML Kata

My personal collection of machine learning katas.

## Description

These katas are intended to be completable within an hour without external assistance. A core requirement is that they be completely unit testable. The goal of these katas is gain the ability to perform techniques without hesitation.

There is no meaning behind the kata order; they are simply in the order in which I wrote them.

## How to

To practice a kata:
1. Navigate to the kata folder and read the `README.md`.
2. Delete the relevant implementation files.
3. Reimplement the required files.
4. Do not use external assistance (e.g. internet). It is fair game to read the unit tests.
4. Run the unit tests via `python -m unittest kataX.XXX_test`.
5. Compare your implementation against the reference afterwards.

## List of Katas

1. **MLP Gradients**. Compute forward/backward for linear, relu, and log_softmax.
2. **MLP Digits**. Code a simple train loop for digits (MLP, dataset, optimizer).
3. **Logistic SGD**. Code multinomial logistic regression training from scratch with SGD.
4. **RNN**. Write a 2-layer Elman RNN with tanh non-linearity.
5. **Scaled Dot Product Attention.** Implement scaled dot product attention.
6. **Optimizers**. Implement SGD with momentum, Adagrad, RMSprop, Adam, AdamW.