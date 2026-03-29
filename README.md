![Tests](https://github.com/jeffdshen/ml-kata/actions/workflows/test.yml/badge.svg)
![Python 3.10](https://img.shields.io/badge/python-3.10-blue)

# ML Kata

My personal collection of machine learning katas.

## Prerequisites

- [uv](https://docs.astral.sh/uv/)
- Python 3.10

Install dependencies:
```bash
uv sync
```

## Description

These katas are intended to be completable within an hour without external assistance. A core requirement is that they be completely unit testable. The goal of these katas is gain the ability to perform techniques without hesitation.

There is no meaning behind the kata order; they are simply in the order in which I wrote them.

## How to

To practice a kata:
1. Navigate to the kata folder and read the `README.md`.
2. Implement the required files.
3. Do not use external assistance (e.g. internet). It is fair game to read the unit tests. Do not look at the `sol` subdirectories.
4. Run the unit tests via `uv run python -m unittest discover -s kataXX -p "*_test.py"` or a specific test via `uv run python -m unittest kataXX.XXX_test`.
5. Compare your implementation against the reference in `sol/` afterwards.

To run tests against the reference solutions, set `ML_KATA_SOL=1`:
```bash
ML_KATA_SOL=1 uv run python -m unittest discover -s kataXX -p "*_test.py"
```

## List of Katas

| # | Name | Description |
|---|------|-------------|
| 1 | MLP Gradients | Compute forward/backward for linear, relu, and log_softmax |
| 2 | MLP Digits | Code a simple train loop for digits (MLP, dataset, optimizer) |
| 3 | Logistic SGD | Code multinomial logistic regression training from scratch with SGD |
| 4 | RNN | Write a 2-layer Elman RNN with tanh non-linearity |
| 5 | Scaled Dot Product Attention | Implement scaled dot product attention |
| 6 | Optimizers | Implement SGD with momentum, Adagrad, RMSprop, Adam, AdamW |
| 7 | Multi-head Attention | Implement multi-head attention |
| 8 | Softmax | Compute forward/backward for softmax |
| 9 | RoBERTa | Implement RoBERTa from scratch |
| 10 | GPT-2 | Implement a GPT-2 transformer decoder model from scratch |
| 11 | Llama | Implement a Llama transformer decoder model from scratch |
| 12 | Reinforcement Learning | WIP |
| 13 | Mixture of Experts | Implement Mixture of Experts (MoE) components |