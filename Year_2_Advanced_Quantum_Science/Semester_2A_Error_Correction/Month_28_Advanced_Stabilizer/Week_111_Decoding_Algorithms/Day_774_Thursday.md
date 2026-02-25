# Day 774: Neural Network Decoders

## Week 111: Decoding Algorithms | Month 28: Advanced Stabilizer Codes

---

## Daily Schedule

| Session | Time | Duration | Focus |
|---------|------|----------|-------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Decoding as Classification & CNN Architectures |
| Afternoon | 1:00 PM - 4:00 PM | 3 hours | Training Data Generation & RNN Approaches |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab: Neural Decoder Implementation |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of this day, you will be able to:

1. **Formulate** quantum decoding as a supervised classification problem
2. **Design** CNN architectures that exploit syndrome structure
3. **Generate** training data through Monte Carlo error simulation
4. **Implement** RNN decoders for temporal syndrome sequences
5. **Analyze** neural decoder thresholds and compare to classical decoders
6. **Evaluate** trade-offs between accuracy, latency, and hardware requirements

---

## Core Content

### 1. Decoding as Supervised Learning

The insight enabling neural decoders is that decoding can be framed as **pattern recognition**:

**Classification Problem:**
- **Input**: Syndrome bit string $s \in \{0, 1\}^m$
- **Output**: Correction class (e.g., logical error type)
- **Training**: Pairs $(s_i, c_i)$ from simulated errors

**Key Observations:**
1. Low-weight errors dominate the posterior
2. Syndrome patterns have spatial structure (for topological codes)
3. Neural networks excel at learning complex pattern mappings

**Output Encoding Options:**

| Approach | Output | Complexity |
|----------|--------|------------|
| Direct | $4^k$ classes (all logical corrections) | Exponential in $k$ |
| Factored | $k$ binary outputs (per logical qubit) | Linear in $k$ |
| Error class | Distinguish $X$, $Z$, $Y$ logical errors | Constant |

For single logical qubit ($k=1$): 4 classes (I, X, Y, Z).

### 2. Convolutional Neural Networks for Surface Codes

Surface code syndromes have **2D spatial structure**, making CNNs natural:

**Input Representation:**
For a distance-$d$ surface code:
- X-syndrome: $(d-1) \times d$ array (or similar)
- Z-syndrome: $d \times (d-1)$ array
- Combined: Treat as 2-channel image

**CNN Architecture:**

```
Input: [d-1, d, 2] syndrome tensor
    |
Conv2D(32 filters, 3x3, ReLU)
    |
Conv2D(64 filters, 3x3, ReLU)
    |
MaxPool2D(2x2)
    |
Conv2D(128 filters, 3x3, ReLU)
    |
GlobalAveragePooling
    |
Dense(256, ReLU)
    |
Dense(4, Softmax)  # Output: I, X, Y, Z probabilities
```

**Why CNNs Work:**
1. **Translation equivariance**: Same error pattern anywhere gives similar features
2. **Hierarchical features**: Low-level (single defects) to high-level (chains)
3. **Parameter sharing**: Fewer parameters than fully connected

### 3. Training Data Generation

Neural decoders require extensive training data generated via Monte Carlo:

**Data Generation Algorithm:**

```python
def generate_training_data(code, error_rate, n_samples):
    data = []
    for _ in range(n_samples):
        # Sample random error
        error = sample_pauli_error(code.n_qubits, error_rate)

        # Compute syndrome
        syndrome = code.measure_syndrome(error)

        # Compute label (logical error class)
        label = code.logical_error_class(error)

        data.append((syndrome, label))
    return data
```

**Training Considerations:**

1. **Class imbalance**: Low error rates mean most samples are "no error"
   - Solution: Oversample error cases or use weighted loss

2. **Error rate curriculum**: Train at various error rates
   - Start with high $p$, gradually decrease

3. **Degeneracy handling**: Multiple errors $\to$ same correction
   - Label by equivalence class, not specific error

**Data Volume Requirements:**

| Code Distance | Syndrome Bits | Training Samples | Training Time |
|---------------|---------------|------------------|---------------|
| 5 | ~24 | $10^5$ | Minutes |
| 9 | ~80 | $10^6$ | Hours |
| 17 | ~288 | $10^7$ | Days |

### 4. Recurrent Neural Networks for Temporal Decoding

With measurement errors, syndromes arrive as **temporal sequences**:

$$s^{(1)}, s^{(2)}, \ldots, s^{(T)}$$

RNNs process this sequence to decode:

**LSTM Architecture:**

```
Input: Sequence of T syndrome frames
    |
LSTM(hidden_size=128, return_sequences=True)
    |
LSTM(hidden_size=128)
    |
Dense(64, ReLU)
    |
Dense(4, Softmax)
```

**Attention Mechanisms:**
Self-attention can identify correlations between distant syndrome rounds:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Transformer Decoder:**
Recent work applies transformers to syndrome sequences, achieving competitive thresholds with excellent parallelization.

### 5. Threshold Analysis

Neural decoders achieve thresholds competitive with or exceeding classical decoders:

**Reported Thresholds (approximate):**

| Decoder | Threshold | Notes |
|---------|-----------|-------|
| MLD (optimal) | 10.9% | Intractable |
| MWPM | 10.3% | $O(n^3)$ |
| Union-Find | 9.9% | $O(n \alpha(n))$ |
| CNN (basic) | ~10.0% | $O(n)$ inference |
| CNN (deep) | ~10.3% | Matches MWPM |
| RNN (temporal) | ~10.2% | Handles measurement errors |

**Factors Affecting Neural Threshold:**
1. Network architecture depth and width
2. Training data quality and quantity
3. Error model match (train/test distribution)
4. Noise in real syndromes vs. simulation

### 6. Real-Time Inference Constraints

Neural decoders must meet strict latency requirements:

**Latency Analysis:**

For distance-$d$ code:
- Syndrome extraction: $O(1)$ per round (~1 $\mu$s)
- Decoder latency must be: $< $ syndrome period
- Backlog accumulates if decoder slower than syndrome rate

**Neural Network Inference Time:**

| Hardware | Small CNN | Large CNN |
|----------|-----------|-----------|
| CPU | ~1 ms | ~10 ms |
| GPU | ~0.1 ms | ~1 ms |
| FPGA | ~10 $\mu$s | ~100 $\mu$s |
| ASIC | ~1 $\mu$s | ~10 $\mu$s |

**Optimizations for Real-Time:**
1. **Quantization**: Use INT8 or binary weights
2. **Pruning**: Remove redundant connections
3. **Knowledge distillation**: Train small network from large
4. **Hardware acceleration**: Custom FPGA/ASIC implementations

### 7. Advantages and Disadvantages

**Advantages of Neural Decoders:**

| Advantage | Explanation |
|-----------|-------------|
| Adaptability | Can learn from real hardware data |
| Correlated noise | Naturally handles complex noise models |
| Parallelism | GPU/TPU acceleration |
| Fixed inference time | $O(n)$ with constant factors |

**Disadvantages:**

| Disadvantage | Explanation |
|--------------|-------------|
| Training cost | Requires extensive data generation |
| Generalization | May not transfer across devices |
| Interpretability | "Black box" decisions |
| Hardware overhead | May require GPU for real-time |

---

## Worked Examples

### Example 1: CNN Input Encoding

**Problem:** Encode the syndrome of a distance-5 surface code for CNN input.

**Solution:**

Distance-5 rotated surface code has:
- 12 X-stabilizers arranged in a $3 \times 4$ pattern
- 12 Z-stabilizers arranged in a $4 \times 3$ pattern

**Syndrome Tensor Construction:**

```python
def encode_syndrome(x_syndrome, z_syndrome, d=5):
    """
    Encode surface code syndrome as CNN input tensor.

    Args:
        x_syndrome: List of 12 X-stabilizer outcomes
        z_syndrome: List of 12 Z-stabilizer outcomes

    Returns:
        Tensor of shape [4, 4, 2]
    """
    # X-syndrome channel
    x_channel = np.zeros((4, 4))
    x_channel[:3, :4] = np.array(x_syndrome).reshape(3, 4)

    # Z-syndrome channel
    z_channel = np.zeros((4, 4))
    z_channel[:4, :3] = np.array(z_syndrome).reshape(4, 3)

    # Stack into 2-channel tensor
    return np.stack([x_channel, z_channel], axis=-1)
```

For syndrome $s_X = [1,0,0,0,0,1,0,0,0,0,0,0]$, $s_Z = [0,0,1,0,0,0,0,0,0,0,0,0]$:

```
X-channel:          Z-channel:
[[1,0,0,0],         [[0,0,1,0],
 [0,1,0,0],          [0,0,0,0],
 [0,0,0,0],          [0,0,0,0],
 [0,0,0,0]]          [0,0,0,0]]
```

### Example 2: Training Data Imbalance

**Problem:** For error rate $p = 0.01$ on a 25-qubit code, calculate the class distribution.

**Solution:**

Probability of no error (identity correction):
$$P(\text{no error}) = (1-p)^{25} = 0.99^{25} \approx 0.778$$

Probability of exactly one error:
$$P(\text{weight-1}) = \binom{25}{1} p (1-p)^{24} \cdot 3 = 75 \cdot 0.01 \cdot 0.99^{24} \approx 0.196$$

Probability of weight-2 or higher:
$$P(\text{weight} \geq 2) \approx 0.026$$

**Class Distribution:**
- ~78% "no correction needed"
- ~20% single-qubit corrections
- ~2% multi-qubit corrections

**Mitigation Strategies:**
1. **Oversampling**: Generate more high-weight error samples
2. **Weighted loss**: Penalize misclassification of rare classes more
3. **Conditional training**: Separate networks for different error weights

### Example 3: Latency Budget Calculation

**Problem:** A quantum computer operates at 1 MHz syndrome rate. How fast must the neural decoder be to avoid backlog at $p = 0.1\%$?

**Solution:**

**Syndrome rate:** 1 MHz = 1 syndrome per $\mu$s

**Decoder requirement:** Must process syndromes faster than they arrive (on average).

**With error probability $p = 0.001$:**
- Trivial syndromes (all zeros): 99.9% of cases
- These require fast path (near-instant)

**For non-trivial syndromes:**
Let $t_{\text{decode}}$ be the average decode time.

Backlog stability condition:
$$\text{Rate}_{\text{in}} < \text{Rate}_{\text{out}}$$
$$1 \text{ MHz} < \frac{1}{t_{\text{decode}}}$$

So $t_{\text{decode}} < 1 \mu$s on average.

**Practical implementation:**
- Fast path for trivial syndromes: $< 0.1 \mu$s
- Full neural inference for non-trivial: $< 10 \mu$s
- Average: $0.999 \times 0.1 + 0.001 \times 10 = 0.11 \mu$s

This is achievable with FPGA implementation.

---

## Practice Problems

### Level A: Direct Application

**A1.** For a distance-3 surface code with 4 X-stabilizers and 4 Z-stabilizers, design the input tensor shape for a CNN decoder.

**A2.** Calculate the number of training samples needed to see each syndrome at least 10 times on average for a code with 16 syndrome bits.

**A3.** A CNN has 50,000 parameters. Estimate the memory required for FP32, FP16, and INT8 representations.

### Level B: Intermediate Analysis

**B1.** Design a loss function that handles the class imbalance problem when 99% of syndromes are trivial. Include both cross-entropy and a reweighting scheme.

**B2.** Compare the receptive field of a 3-layer CNN (3x3 kernels) with the maximum distance between correlated defects in a distance-9 surface code. Is the receptive field sufficient?

**B3.** Analyze how neural decoder accuracy changes when trained on depolarizing noise but tested on biased noise (10:1 Z:X ratio).

### Level C: Advanced Problems

**C1.** Prove that a sufficiently deep neural network can represent the optimal MLD decoder for any stabilizer code. What is the minimum depth required?

**C2.** Design a neural decoder architecture that maintains equivariance under lattice symmetries (rotations, reflections) of the surface code.

**C3.** Develop a transfer learning approach that adapts a neural decoder trained on simulated data to a real quantum device with unknown noise characteristics using only 1000 real syndrome samples.

---

## Computational Lab: Neural Decoder Implementation

```python
"""
Day 774 Computational Lab: Neural Network Decoders
Training and evaluating neural decoders for surface codes

This lab implements CNN and simple RNN decoders, demonstrating
training, evaluation, and comparison with classical decoders.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from dataclasses import dataclass
import time

# Neural network imports (using numpy for portability)
# In practice, use TensorFlow or PyTorch


class SurfaceCodeSimulator:
    """Simulates surface code errors and syndromes for training data."""

    def __init__(self, distance: int):
        self.d = distance
        self.n_data = distance ** 2
        self.n_x_stab = (distance - 1) * distance
        self.n_z_stab = distance * (distance - 1)
        self.n_syndrome = self.n_x_stab + self.n_z_stab

    def sample_error(self, p: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample random Pauli error.

        Returns:
            (x_errors, z_errors) each of shape (n_data,)
        """
        # Sample X and Z errors independently (depolarizing-like)
        x_errors = np.random.random(self.n_data) < p
        z_errors = np.random.random(self.n_data) < p
        return x_errors.astype(np.float32), z_errors.astype(np.float32)

    def compute_syndrome(self, x_errors: np.ndarray,
                        z_errors: np.ndarray) -> np.ndarray:
        """
        Compute syndrome from error pattern.

        Returns:
            Syndrome array of shape (n_syndrome,)
        """
        # Simplified syndrome computation
        x_syndrome = np.zeros(self.n_x_stab, dtype=np.float32)
        z_syndrome = np.zeros(self.n_z_stab, dtype=np.float32)

        # X-stabilizers detect Z errors
        error_grid = z_errors.reshape(self.d, self.d)
        idx = 0
        for i in range(self.d - 1):
            for j in range(self.d):
                if j < self.d - 1:
                    x_syndrome[idx] = (error_grid[i, j] + error_grid[i+1, j] +
                                       error_grid[i, j+1] + error_grid[i+1, j+1]) % 2
                idx += 1

        # Z-stabilizers detect X errors
        error_grid = x_errors.reshape(self.d, self.d)
        idx = 0
        for i in range(self.d):
            for j in range(self.d - 1):
                if i < self.d - 1:
                    z_syndrome[idx] = (error_grid[i, j] + error_grid[i+1, j] +
                                       error_grid[i, j+1] + error_grid[i+1, j+1]) % 2
                idx += 1

        return np.concatenate([x_syndrome, z_syndrome])

    def compute_logical_error(self, x_errors: np.ndarray,
                             z_errors: np.ndarray) -> int:
        """
        Compute logical error class: 0=I, 1=X, 2=Z, 3=Y

        Simplified: check parity along logical operators.
        """
        x_grid = x_errors.reshape(self.d, self.d)
        z_grid = z_errors.reshape(self.d, self.d)

        # Logical X: odd X parity along a column
        log_x = np.sum(x_grid[:, self.d // 2]) % 2

        # Logical Z: odd Z parity along a row
        log_z = np.sum(z_grid[self.d // 2, :]) % 2

        return int(log_x + 2 * log_z)


def generate_dataset(simulator: SurfaceCodeSimulator,
                    n_samples: int,
                    error_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate training dataset.

    Returns:
        (syndromes, labels) arrays
    """
    syndromes = []
    labels = []

    for _ in range(n_samples):
        x_err, z_err = simulator.sample_error(error_rate)
        syndrome = simulator.compute_syndrome(x_err, z_err)
        label = simulator.compute_logical_error(x_err, z_err)

        syndromes.append(syndrome)
        labels.append(label)

    return np.array(syndromes), np.array(labels)


class SimpleNeuralDecoder:
    """
    Simple feedforward neural network decoder.

    Uses numpy for portability. In practice, use PyTorch/TensorFlow.
    """

    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int = 4):
        """
        Initialize neural network.

        Args:
            input_size: Number of syndrome bits
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output classes
        """
        self.layers = []
        sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(sizes) - 1):
            # Xavier initialization
            scale = np.sqrt(2.0 / (sizes[i] + sizes[i+1]))
            W = np.random.randn(sizes[i], sizes[i+1]) * scale
            b = np.zeros(sizes[i+1])
            self.layers.append({'W': W, 'b': b})

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through network."""
        self.activations = [x]

        for i, layer in enumerate(self.layers[:-1]):
            x = x @ layer['W'] + layer['b']
            x = np.maximum(0, x)  # ReLU
            self.activations.append(x)

        # Output layer (softmax)
        x = x @ self.layers[-1]['W'] + self.layers[-1]['b']
        x = self.softmax(x)
        self.activations.append(x)

        return x

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probs = self.forward(x)
        return np.argmax(probs, axis=-1)

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute cross-entropy loss."""
        probs = self.forward(x)
        n = len(y)
        # Add small epsilon for numerical stability
        log_probs = np.log(probs[np.arange(n), y] + 1e-10)
        return -np.mean(log_probs)

    def train_step(self, x: np.ndarray, y: np.ndarray,
                   learning_rate: float = 0.01) -> float:
        """
        Single training step with backpropagation.

        Returns:
            Loss value
        """
        # Forward pass
        probs = self.forward(x)
        n = len(y)

        # Compute loss
        loss = -np.mean(np.log(probs[np.arange(n), y] + 1e-10))

        # Backward pass (simplified gradient descent)
        # Output layer gradient
        grad = probs.copy()
        grad[np.arange(n), y] -= 1
        grad /= n

        # Backpropagate through layers
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            activation = self.activations[i]

            # Weight gradient
            dW = activation.T @ grad
            db = np.sum(grad, axis=0)

            # Propagate gradient
            if i > 0:
                grad = grad @ layer['W'].T
                # ReLU gradient
                grad = grad * (self.activations[i] > 0)

            # Update weights
            layer['W'] -= learning_rate * dW
            layer['b'] -= learning_rate * db

        return loss

    def train(self, x_train: np.ndarray, y_train: np.ndarray,
              epochs: int = 100, batch_size: int = 64,
              learning_rate: float = 0.01, verbose: bool = True) -> List[float]:
        """
        Train the network.

        Returns:
            List of loss values per epoch
        """
        losses = []
        n = len(x_train)

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0
            n_batches = 0

            # Mini-batch training
            for i in range(0, n, batch_size):
                x_batch = x_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                batch_loss = self.train_step(x_batch, y_batch, learning_rate)
                epoch_loss += batch_loss
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        return losses


class ConvNeuralDecoder:
    """
    Simplified CNN decoder using manual convolution.

    For production, use PyTorch/TensorFlow Conv2D layers.
    """

    def __init__(self, input_shape: Tuple[int, int, int],
                 n_filters: int = 16, output_size: int = 4):
        self.input_shape = input_shape
        h, w, c = input_shape

        # Simple 3x3 convolution filters
        self.conv_filters = np.random.randn(3, 3, c, n_filters) * 0.1
        self.conv_bias = np.zeros(n_filters)

        # Dense layer after flattening
        conv_out_size = (h - 2) * (w - 2) * n_filters
        self.dense_W = np.random.randn(conv_out_size, output_size) * 0.1
        self.dense_b = np.zeros(output_size)

    def conv2d(self, x: np.ndarray) -> np.ndarray:
        """Apply convolution."""
        batch, h, w, c = x.shape
        n_filters = self.conv_filters.shape[-1]

        out_h, out_w = h - 2, w - 2
        output = np.zeros((batch, out_h, out_w, n_filters))

        for i in range(out_h):
            for j in range(out_w):
                patch = x[:, i:i+3, j:j+3, :]  # [batch, 3, 3, c]
                for f in range(n_filters):
                    output[:, i, j, f] = np.sum(
                        patch * self.conv_filters[:, :, :, f],
                        axis=(1, 2, 3)
                    ) + self.conv_bias[f]

        return output

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        # Reshape to image format
        batch = x.shape[0]
        h, w, c = self.input_shape
        x = x.reshape(batch, h, w, c)

        # Convolution + ReLU
        x = self.conv2d(x)
        x = np.maximum(0, x)

        # Flatten
        x = x.reshape(batch, -1)

        # Dense layer
        x = x @ self.dense_W + self.dense_b

        # Softmax
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probs = self.forward(x)
        return np.argmax(probs, axis=-1)


def evaluate_decoder(decoder, x_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """
    Evaluate decoder performance.

    Returns:
        Dictionary with accuracy metrics
    """
    predictions = decoder.predict(x_test)
    accuracy = np.mean(predictions == y_test)

    # Per-class accuracy
    class_accuracy = {}
    for c in range(4):
        mask = y_test == c
        if np.sum(mask) > 0:
            class_accuracy[c] = np.mean(predictions[mask] == c)
        else:
            class_accuracy[c] = None

    return {
        'accuracy': accuracy,
        'class_accuracy': class_accuracy,
        'predictions': predictions
    }


def train_and_evaluate():
    """
    Main training and evaluation pipeline.
    """
    print("=" * 60)
    print("Neural Decoder Training Pipeline")
    print("=" * 60)

    # Setup
    distance = 5
    simulator = SurfaceCodeSimulator(distance)
    error_rate = 0.08

    print(f"\nCode distance: {distance}")
    print(f"Syndrome bits: {simulator.n_syndrome}")
    print(f"Error rate: {error_rate}")

    # Generate datasets
    print("\nGenerating training data...")
    x_train, y_train = generate_dataset(simulator, n_samples=20000, error_rate=error_rate)
    x_test, y_test = generate_dataset(simulator, n_samples=5000, error_rate=error_rate)

    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Class distribution: {np.bincount(y_train)}")

    # Train feedforward network
    print("\n" + "-" * 40)
    print("Training Feedforward Neural Decoder")
    print("-" * 40)

    ff_decoder = SimpleNeuralDecoder(
        input_size=simulator.n_syndrome,
        hidden_sizes=[128, 64, 32]
    )

    start_time = time.time()
    losses = ff_decoder.train(x_train, y_train, epochs=50,
                             batch_size=128, learning_rate=0.01)
    train_time = time.time() - start_time

    print(f"\nTraining time: {train_time:.2f} seconds")

    # Evaluate
    results_ff = evaluate_decoder(ff_decoder, x_test, y_test)
    print(f"Test accuracy: {results_ff['accuracy']:.4f}")
    print(f"Per-class accuracy: {results_ff['class_accuracy']}")

    # Measure inference time
    start_time = time.time()
    for _ in range(100):
        _ = ff_decoder.predict(x_test[:100])
    inference_time = (time.time() - start_time) / 100 / 100  # per sample

    print(f"Inference time: {inference_time*1e6:.2f} μs per syndrome")

    return losses, results_ff


def threshold_estimation():
    """
    Estimate neural decoder threshold.
    """
    print("\n" + "=" * 60)
    print("Neural Decoder Threshold Estimation")
    print("=" * 60)

    distances = [3, 5, 7]
    error_rates = np.linspace(0.05, 0.15, 11)
    n_test = 2000

    results = {d: [] for d in distances}

    for d in distances:
        print(f"\nDistance {d}:")
        simulator = SurfaceCodeSimulator(d)

        # Train decoder at moderate error rate
        x_train, y_train = generate_dataset(simulator, 10000, error_rate=0.10)

        decoder = SimpleNeuralDecoder(
            input_size=simulator.n_syndrome,
            hidden_sizes=[64, 32]
        )
        decoder.train(x_train, y_train, epochs=30, verbose=False)

        # Test at various error rates
        for p in error_rates:
            x_test, y_test = generate_dataset(simulator, n_test, error_rate=p)
            res = evaluate_decoder(decoder, x_test, y_test)

            # Logical error rate = 1 - accuracy (simplified)
            logical_error_rate = 1 - res['accuracy']
            results[d].append(logical_error_rate)

            print(f"  p={p:.2f}: accuracy={res['accuracy']:.4f}, "
                  f"logical_error={logical_error_rate:.4f}")

    # Plot results
    plt.figure(figsize=(10, 7))

    colors = ['blue', 'green', 'red']
    markers = ['o', 's', '^']

    for i, d in enumerate(distances):
        plt.semilogy(error_rates * 100, results[d],
                    f'{colors[i]}{markers[i]}-',
                    label=f'd = {d}', linewidth=2, markersize=8)

    plt.axvline(x=10.0, color='orange', linestyle='--',
                label='Approx. neural threshold (~10.0%)')
    plt.axvline(x=10.3, color='purple', linestyle=':',
                label='MWPM threshold (~10.3%)')

    plt.xlabel('Physical Error Rate (%)', fontsize=12)
    plt.ylabel('Logical Error Rate', fontsize=12)
    plt.title('Neural Decoder Threshold Estimation', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([5, 15])

    plt.tight_layout()
    plt.savefig('neural_decoder_threshold.png', dpi=150)
    plt.show()


def compare_architectures():
    """
    Compare different neural architectures.
    """
    print("\n" + "=" * 60)
    print("Architecture Comparison")
    print("=" * 60)

    distance = 5
    simulator = SurfaceCodeSimulator(distance)

    # Generate data
    x_train, y_train = generate_dataset(simulator, 15000, error_rate=0.08)
    x_test, y_test = generate_dataset(simulator, 3000, error_rate=0.08)

    architectures = [
        ("Small (32)", [32]),
        ("Medium (64, 32)", [64, 32]),
        ("Large (128, 64, 32)", [128, 64, 32]),
        ("Deep (64, 64, 64, 64)", [64, 64, 64, 64]),
    ]

    results = []

    for name, hidden_sizes in architectures:
        print(f"\nTraining {name}...")

        decoder = SimpleNeuralDecoder(
            input_size=simulator.n_syndrome,
            hidden_sizes=hidden_sizes
        )

        start = time.time()
        decoder.train(x_train, y_train, epochs=40, verbose=False)
        train_time = time.time() - start

        res = evaluate_decoder(decoder, x_test, y_test)

        # Count parameters
        n_params = sum(
            layer['W'].size + layer['b'].size
            for layer in decoder.layers
        )

        results.append({
            'name': name,
            'accuracy': res['accuracy'],
            'n_params': n_params,
            'train_time': train_time
        })

        print(f"  Accuracy: {res['accuracy']:.4f}")
        print(f"  Parameters: {n_params}")
        print(f"  Train time: {train_time:.2f}s")

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    names = [r['name'] for r in results]
    x_pos = range(len(names))

    # Accuracy
    axes[0].bar(x_pos, [r['accuracy'] for r in results], color='steelblue')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(names, rotation=45, ha='right')
    axes[0].set_ylabel('Test Accuracy')
    axes[0].set_title('Accuracy by Architecture')
    axes[0].set_ylim([0.5, 1.0])

    # Parameters
    axes[1].bar(x_pos, [r['n_params'] for r in results], color='coral')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(names, rotation=45, ha='right')
    axes[1].set_ylabel('Number of Parameters')
    axes[1].set_title('Model Size')

    # Training time
    axes[2].bar(x_pos, [r['train_time'] for r in results], color='mediumseagreen')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(names, rotation=45, ha='right')
    axes[2].set_ylabel('Training Time (s)')
    axes[2].set_title('Training Efficiency')

    plt.tight_layout()
    plt.savefig('architecture_comparison.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    # Run demonstrations
    losses, results = train_and_evaluate()

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses, 'b-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.title('Neural Decoder Training Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=150)
    plt.show()

    compare_architectures()
    threshold_estimation()

    print("\n" + "=" * 60)
    print("Lab Complete!")
    print("=" * 60)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Classification Output | $$\hat{c} = \text{argmax}_c \, P(c \vert s; \theta)$$ |
| Cross-Entropy Loss | $$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \log P(c_i \vert s_i; \theta)$$ |
| Softmax | $$P(c \vert s) = \frac{e^{z_c}}{\sum_{c'} e^{z_{c'}}}$$ |
| Neural Threshold | $$p_{\text{th}}^{\text{neural}} \approx 10.0\%$$ |
| Inference Complexity | $$O(n)$$ per forward pass |
| Training Samples | $$N \sim 10^5 \text{ to } 10^7$$ |

### Key Takeaways

1. **Decoding is classification**: Syndrome $\to$ logical correction class
2. **CNNs exploit spatial structure**: Natural for 2D syndrome patterns
3. **Training data is critical**: Requires careful generation and class balancing
4. **Thresholds are competitive**: Neural decoders can match MWPM (~10.0-10.3%)
5. **Real-time requires optimization**: Quantization, pruning, hardware acceleration
6. **Adaptability is a strength**: Can learn from real device data

---

## Daily Checklist

- [ ] Understood decoding as supervised classification
- [ ] Designed CNN architecture for syndrome tensors
- [ ] Implemented training data generation pipeline
- [ ] Trained and evaluated neural decoder
- [ ] Analyzed threshold performance
- [ ] Compared different architectures
- [ ] Completed practice problems (at least Level A and B)

---

## Preview: Day 775

Tomorrow we study **Belief Propagation and LDPC Decoding**, the message-passing algorithms that power classical error correction and are increasingly important for quantum LDPC codes. We'll learn about factor graphs, the sum-product algorithm, and why BP struggles with short cycles in quantum codes.

Key questions for tomorrow:
- How does belief propagation represent correlations in factor graphs?
- What is the min-sum vs sum-product algorithm trade-off?
- Can BP achieve good performance for quantum LDPC codes?

---

*Day 774 of 2184 | Week 111 | Month 28 | Year 2: Advanced Quantum Science*
