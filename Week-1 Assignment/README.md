# Neural Network Fundamentals: Backpropagation from Scratch and with PyTorch

This repository contains solutions to two problems that explore the fundamentals of training neural networks. The solutions demonstrate:
1.  How to perform backpropagation manually using Python and NumPy.
2.  How to leverage PyTorch's automatic differentiation to simplify the training process.
3.  How to build and train a multi-layer perceptron (MLP) using PyTorch's `nn` module.

## Problem 1: Single Neuron with Leaky ReLU and L2 Loss

The goal of this problem is to train a single neuron with two inputs (`x1`, `x2`) and two weights (`w1`, `w2`) to produce an output value of **5**.

-   **Activation Function**: Leaky ReLU
-   **Loss Function**: L2 Loss (Mean Squared Error)

$$ L = (\text{LeakyReLU}(w_1 \cdot x_1 + w_2 \cdot x_2) - 5)^2 $$

### Solution 1a: Backpropagation by Hand

This solution calculates the gradients with respect to the weights (`w1`, `w2`) manually using the **chain rule**. The weights are then updated in a simple training loop. This approach highlights the core mechanics of how a neural network learns.

### Solution 1b: Reimplementing with PyTorch

This part solves the same problem but uses PyTorch tensors and its automatic differentiation engine. By calling `loss.backward()`, PyTorch automatically computes the gradients, which are then used to update the weights manually. This removes the need for manual gradient derivation.

### Solution 1c: Building a 2-Layer Neural Network

This solution builds a complete two-layer neural network using PyTorch's `nn.Module`.
-   **Input Layer**: 32 features
-   **Hidden Layer**: 8 neurons with Leaky ReLU activation
-   **Output Layer**: 1 neuron (no activation)
-   **Optimizer**: Stochastic Gradient Descent (`optim.SGD`)
-   **Loss**: Mean Squared Error (`nn.MSELoss`)

The network is trained on a batch of 4 samples to output the target value of 5.

***

## Problem 2: Single Neuron with Sigmoid and L1 Loss

The second problem shifts to a new objective: training a single neuron with three inputs (`x1`, `x2`, `x3`) to produce an output value of **0.6**.

-   **Activation Function**: Sigmoid
-   **Loss Function**: L1 Loss (Absolute Error)

$$ L = |\text{Sigmoid}(w_1 \cdot x_1 + w_2 \cdot x_2 + w_3 \cdot x_3) - 0.6| $$

### Solution 2a: Backpropagation by Hand

Similar to 1a, this solution uses the chain rule to manually derive the gradients for the weights (`w1`, `w2`, `w3`). The key differences are the derivatives for the Sigmoid activation and the L1 loss function.

### Solution 2b: Reimplementing with PyTorch

This part uses PyTorch's automatic differentiation to solve the same problem. The code demonstrates how easily PyTorch handles different activation (`torch.sigmoid`) and loss (`torch.abs`) functions.

### Solution 2c: Building a 2-Layer Neural Network

This solution constructs another two-layer MLP.
-   **Input Layer**: 32 features
-   **Hidden Layer**: 8 neurons with Sigmoid activation
-   **Output Layer**: 1 neuron with Sigmoid activation
-   **Optimizer**: Stochastic Gradient Descent (`optim.SGD`)
-   **Loss**: Mean Squared Error (`nn.MSELoss`)

The network is trained on a batch of 4 samples to output the target value of 0.6. The final Sigmoid activation ensures the output is always between 0 and 1, which is suitable for this target.