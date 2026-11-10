
# Lab 2: A Neural Network from Scratch

## 1. Goal

Implement a two-hidden-layer neural network from scratch using NumPy.
Verify backpropagation using numerical gradients, compare different
learning rates and hidden-layer widths, and measure forward- and
backward-pass execution times.

## 2. Method

### Dataset

- Dataset: sklearn digits dataset
- Number of classes: 10
- Input features: 64
- Input scaling: divide pixel values by 16
- Train/test split: stratified 70/30
- Random state: 42

### Architecture

64 -> W -> ReLU -> W -> ReLU -> 10 -> Softmax

Hidden widths: 16 and 64.

Weights are initialized from a normal distribution with mean 0
and standard deviation 0.01. Biases are initialized to zero.

### Training configuration

- Epochs: 50
- Batch size: 32
- Learning rates: 0.001, 0.01, 0.1
- Loss: cross-entropy
- Optimizer: mini-batch gradient descent

### Gradient verification

Ten weight elements were tested using centered finite differences
with epsilon = 1e-5.

## 3. Results

### Gradient check

- Passing weights: [fill in]/10
- Maximum relative error: [fill in]
- Overall check: [PASS/FAIL]

### Training and test results

Insert the values from results.csv.

| Width | Learning rate | Final loss | Test accuracy |
|---:|---:|---:|---:|
| 16 | 0.001 | | |
| 16 | 0.01 | | |
| 16 | 0.1 | | |
| 64 | 0.001 | | |
| 64 | 0.01 | | |
| 64 | 0.1 | | |

### Loss curves

Insert the image results/loss_curves.png.

Describe how the loss changes during training and compare the
learning rates and hidden-layer widths.

### Execution time

| Width | Forward pass (ms) | Backward pass (ms) |
|---:|---:|---:|
| 16 | | |
| 64 | | |

Explain which pass takes longer and how the network width affects
the number of matrix operations.

## 4. Conclusions

1. Explain whether numerical gradient checking supported the
   correctness of your backpropagation implementation.
2. Identify the best-performing configuration using your measured
   test accuracy and loss curves.
3. Compare forward- and backward-pass timing and explain how
   matrix multiplication contributes to the observed timings.
