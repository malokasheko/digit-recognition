# Handwritten Digits Recognition Neural Network
**Alex Eagles Bank — Intelligent Automation Unit — Task 3**

An MLP-based digit classifier trained on MNIST, prototyping the recognition engine for an
automated check/form processing pipeline.

## Contents
- `mnist_digit_recognition.ipynb` — data pipeline, model architecture, training/evaluation,
  and a custom inference function, plus the full research write-up inline.
- `research_writeup.md` — standalone copy of the Part 2 technical research/conceptual analysis
  (Topics A, B).
- `requirements.txt` — Python dependencies.

## Running it
```bash
pip install -r requirements.txt
jupyter notebook mnist_digit_recognition.ipynb
```
The MNIST dataset is downloaded automatically on first run via `torchvision.datasets.MNIST`.

## Summary
- Model: 784 → 256 → 128 → 10 fully connected MLP with ReLU activations and dropout.
- Loss: Cross-Entropy. Optimizer: Adam.
- Includes a standalone `predict_digit()` inference function returning the predicted digit and
  full class-probability confidence scores, simulating inference on an isolated digit crop.
