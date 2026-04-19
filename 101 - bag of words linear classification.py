import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.datasets import make_blobs
import torch
import torch.nn as nn
from einops import reduce

# Set a seed so that the results are consistent.
np.random.seed(3)

## --------------------------- Answer for question 1a in XCS221 assignment 2
# For the tweet **"so so worried about tomorrow"**, count each word's occurrences against the vocabulary:
'''
| Index | Word     | Count |
|-------|----------|-------|
| 0     | about    | 1     |
| 1     | amazing  | 0     |
| 2     | angry    | 0     |
| 3     | day      | 0     |
| 4     | hate     | 0     |
| 5     | love     | 0     |
| 6     | of       | 0     |
| 7     | scared   | 0     |
| 8     | so       | **2** |
| 9     | spiders  | 0     |
| 10    | this     | 0     |
| 11    | tomorrow | 1     |
| 12    | waiting  | 0     |
| 13    | worried  | 1     |

**f(x) = [1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 1]**

The key thing to note is that "so" appears **twice**, so its count is 2, while "about", "tomorrow", and "worried" each appear once.
'''

## --------------------------- Answer for 1b
'''
## Softmax Computation

The softmax formula is:

$$P(i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

**Step 1: Compute the exponentials**

| Class | z    | e^z            |
|-------|------|----------------|
| Joy   | 2.0  | e^2.0 = 7.389  |
| Anger | 1.0  | e^1.0 = 2.718  |
| Fear  | -1.0 | e^-1.0 = 0.368 |

**Step 2: Compute the sum**

$$\sum e^{z_j} = 7.389 + 2.718 + 0.368 = 10.475$$

**Step 3: Divide each by the sum**

$$P(\text{Joy})   = \frac{7.389}{10.475} = 0.705$$

$$P(\text{Anger}) = \frac{2.718}{10.475} = 0.259$$

$$P(\text{Fear})  = \frac{0.368}{10.475} = 0.035$$

**Result: [P(Joy), P(Anger), P(Fear)] = [0.705, 0.259, 0.035]**

As a sanity check, the probabilities sum to 1: 0.705 + 0.259 + 0.035 = 0.999 ✓ (rounding error only)
'''
## --------------------------- Answer for 1c
'''
## Cross-Entropy Loss Computation

The cross-entropy loss formula is:

$$L = -\sum_i y_i \log(p_i)$$

where **y** is the one-hot true label vector and **p** is the predicted probability vector.

**Given:**
- True label: **y** = [0, 1, 0]
- Predicted probabilities: **p** = [0.2, 0.7, 0.1]

**Step 1: Expand the sum**

$$L = -(y_{\text{Joy}}\log p_{\text{Joy}} + y_{\text{Anger}}\log p_{\text{Anger}} + y_{\text{Fear}}\log p_{\text{Fear}})$$

$$L = -(0 \cdot \log(0.2) + 1 \cdot \log(0.7) + 0 \cdot \log(0.1))$$

**Step 2: Because of the one-hot encoding, only the correct class term survives**

$$L = -\log(0.7) = 0.357$$

### **i. Cross-Entropy Loss = 0.357**

---

### **ii. Loss Behavior as P(correct class) → 1**

As the predicted probability for the correct class approaches 1, the loss approaches 0, since $-\log(1) = 0$. 
Conversely, as that probability approaches 0, the loss grows toward **+∞**, since $-\log(x) \to \infty$ as $x \to 0$. 
This asymmetric behavior means the model is **heavily penalized** for being confidently wrong, which encourages 
the model to assign high probability to the true class during training.
'''

## --------------------------- Answer for 1d
'''
## Derivation of ∂Loss_CE / ∂z_k

### Setup

Recall the two components:

$$\text{Loss}_{CE} = -\sum_i y_i \log(p_i) \qquad \text{and} \qquad p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

---

### Step 1: Apply the Chain Rule

$$\frac{\partial L}{\partial z_k} = \sum_i \frac{\partial L}{\partial p_i} \cdot \frac{\partial p_i}{\partial z_k}$$

**Computing ∂L/∂p_i** (straightforward log derivative):

$$\frac{\partial L}{\partial p_i} = -\frac{y_i}{p_i}$$

---

### Step 2: Compute the Softmax Jacobian ∂p_i/∂z_k

This requires two cases:

**Case 1 — i = k** (quotient rule, where numerator and denominator both depend on z_k):

$$\frac{\partial p_k}{\partial z_k} = \frac{e^{z_k}\sum_j e^{z_j} - e^{z_k} \cdot e^{z_k}}{\left(\sum_j e^{z_j}\right)^2} = p_k\left(1 - p_k\right)$$

**Case 2 — i ≠ k** (only denominator depends on z_k):

$$\frac{\partial p_i}{\partial z_k} = \frac{0 - e^{z_i} \cdot e^{z_k}}{\left(\sum_j e^{z_j}\right)^2} = -p_i p_k$$

---

### Step 3: Combine via Chain Rule

$$\frac{\partial L}{\partial z_k} = \underbrace{\left(-\frac{y_k}{p_k}\right) \cdot p_k(1-p_k)}_{\text{i = k term}} + \underbrace{\sum_{i \neq k}\left(-\frac{y_i}{p_i}\right)\cdot(-p_i p_k)}_{\text{i ≠ k terms}}$$

$$= -y_k(1-p_k) + p_k\sum_{i \neq k} y_i$$

$$= -y_k + y_k p_k + p_k \sum_{i \neq k} y_i$$

$$= -y_k + p_k\underbrace{\left(y_k + \sum_{i \neq k} y_i\right)}_{= \sum_i y_i = 1 \text{ (one-hot)}}$$

---

### **ii. Final Gradient Expression**

$$\boxed{\frac{\partial L}{\partial z_k} = p_k - y_k}$$

---

### **iii. Intuitive Explanation**

The gradient is simply the **difference between what the model predicted and what the true label was**. If the 
model over-estimates the probability of class k (p_k > y_k), the gradient is positive and pushes z_k **down**; 
if it under-estimates (p_k < y_k), the gradient is negative and pushes z_k **up**. When the model is perfectly 
correct (p_k = y_k = 1 for the true class), the gradient vanishes to zero — meaning there is **nothing left to 
learn**, which is exactly the desired behavior for gradient-based optimization.
'''

## --------------------------- Answer for 2a
'''
## Averaged Embedding for "so angry"

### Step 1: Look up word embeddings

| Word  | Embedding    |
|-------|--------------|
| so    | [0.1, 0.0]   |
| angry | [-0.6, -0.8] |

### Step 2: Compute the element-wise average

$$f(\text{"so angry"}) = \frac{1}{2}\left([0.1, 0.0] + [-0.6, -0.8]\right)$$

$$= \frac{1}{2}[0.1 + (-0.6),\ 0.0 + (-0.8)]$$

$$= \frac{1}{2}[-0.5,\ -0.8]$$

### **i. Averaged Embedding = [-0.25, -0.4]**

---

### **ii. Advantage & Disadvantage**

**Advantage — Simplicity and generalization:** Averaging is computationally cheap and produces a fixed-size representation regardless of tweet length. It also leverages the semantic geometry of the embedding space — tweets with semantically similar words will have similar average vectors, enabling reasonable generalization even on unseen text.

**Disadvantage — Loss of word order and syntax:** Averaging discards all positional and structural information, making "I am not happy" and "I am happy" potentially very similar in representation. This means the approach cannot capture negation, context, or the relative importance of individual words, which are often critical for accurate sentiment classification.
'''

## --------------------------- Answer for 2b
'''
## Forward Pass for "so angry"

### Input: x

From the previous part, the averaged embedding for "so angry" is:

$$x = \begin{bmatrix} -0.25 \\ -0.4 \end{bmatrix}$$

---

### Step 1: Pre-activation of Hidden Layer → W⁽¹⁾x + b⁽¹⁾

$$W^{(1)}x + b^{(1)} = \begin{bmatrix} 1.0 & 0.5 \\ -0.5 & 1.0 \end{bmatrix} \begin{bmatrix} -0.25 \\ -0.4 \end{bmatrix} + \begin{bmatrix} 0.1 \\ -0.2 \end{bmatrix}$$

**Row 1:** $1.0(-0.25) + 0.5(-0.4) + 0.1 = -0.25 - 0.20 + 0.1 = -0.35$

**Row 2:** $-0.5(-0.25) + 1.0(-0.4) + (-0.2) = 0.125 - 0.40 - 0.2 = -0.475$

$$W^{(1)}x + b^{(1)} = \begin{bmatrix} -0.35 \\ -0.475 \end{bmatrix}$$

---

### Step 2: Apply ReLU Activation → h

$$h = \text{ReLU}\begin{bmatrix} -0.35 \\ -0.475 \end{bmatrix} = \begin{bmatrix} \max(0, -0.35) \\ \max(0, -0.475) \end{bmatrix} = \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}$$

> Both values are negative, so ReLU clamps them both to **0**.

---

### Step 3: Output Pre-activation → z

$$z = W^{(2)}h + b^{(2)} = \begin{bmatrix} 0.8 & -0.6 \end{bmatrix} \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix} + 0.3$$

$$z = 0.8(0.0) + (-0.6)(0.0) + 0.3 = \mathbf{0.3}$$

---

### Step 4: Apply Sigmoid Activation → ŷ

$$\hat{y} = \sigma(0.3) = \frac{1}{1 + e^{-0.3}} = \frac{1}{1 + 0.741} = \frac{1}{1.741} \approx \mathbf{0.574}$$

---

### Summary of Forward Pass

| Variable | Value |
|----------|-------|
| **x** | [-0.25, -0.4]ᵀ |
| **W⁽¹⁾x + b⁽¹⁾** | [-0.35, -0.475]ᵀ |
| **h** | [0.0, 0.0]ᵀ |
| **z** | 0.3 |
| **ŷ** | 0.574 |

Since $\hat{y} = 0.574 > 0.5$, the model predicts **positive sentiment** — which is incorrect for "so angry". This is largely because the strongly negative embeddings caused both ReLU neurons to **die** (output 0), losing all input signal and leaving only the bias b⁽²⁾ = 0.3 to drive the prediction.
'''


## --------------------------- Answer for 2c
'''
## Backpropagation for "so angry"

### Reusing values from the forward pass:
| Variable | Value |
|----------|-------|
| x | [-0.25, -0.4]ᵀ |
| pre-activation | [-0.35, -0.475]ᵀ |
| h | [0.0, 0.0]ᵀ |
| z | 0.3 |
| ŷ | 0.574 |
| y_true | 0 |

---

## i. Computing Loss L

Binary cross-entropy loss:

$$L = -[y_{\text{true}} \log(\hat{y}) + (1 - y_{\text{true}})\log(1-\hat{y})]$$

$$L = -[0 \cdot \log(0.574) + 1 \cdot \log(1 - 0.574)]$$

$$L = -\log(0.426) = \mathbf{0.853}$$

---

## ii. Backward Pass

### ∂L/∂ŷ — Gradient of loss w.r.t. prediction

$$\frac{\partial L}{\partial \hat{y}} = -\left[\frac{y_{\text{true}}}{\hat{y}} - \frac{1 - y_{\text{true}}}{1 - \hat{y}}\right] = \frac{1}{1 - \hat{y}} = \frac{1}{1-0.574} = \frac{1}{0.426} \approx \mathbf{2.347}$$

---

### ∂L/∂z — Through sigmoid (using the clean combined result)

$$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} = \frac{1}{1-\hat{y}} \cdot \hat{y}(1-\hat{y}) = \hat{y} - y_{\text{true}}$$

$$= 0.574 - 0 = \mathbf{0.574}$$

---

### ∂L/∂W⁽²⁾ — Gradient w.r.t. output weights

$$\frac{\partial L}{\partial W^{(2)}} = \frac{\partial L}{\partial z} \cdot h^T = 0.574 \cdot [0.0,\ 0.0] = \mathbf{[0.0,\ 0.0]}$$

---

### ∂L/∂b⁽²⁾ — Gradient w.r.t. output bias

$$\frac{\partial L}{\partial b^{(2)}} = \frac{\partial L}{\partial z} \cdot 1 = \mathbf{0.574}$$

---

### ∂L/∂h — Gradient w.r.t. hidden activations

$$\frac{\partial L}{\partial h} = \frac{\partial L}{\partial z} \cdot W^{(2)T} = 0.574 \cdot \begin{bmatrix} 0.8 \\ -0.6 \end{bmatrix} = \begin{bmatrix} 0.459 \\ -0.344 \end{bmatrix}$$

---

### Through ReLU — Gradient w.r.t. pre-activation

$$\frac{\partial L}{\partial \text{pre-act}} = \frac{\partial L}{\partial h} \odot \text{ReLU}'(\text{pre-act})$$

Since pre-activations were **[-0.35, -0.475]**, both **negative**, ReLU' = 0 for both neurons:

$$= \begin{bmatrix} 0.459 \\ -0.344 \end{bmatrix} \odot \begin{bmatrix} 0 \\ 0 \end{bmatrix} = \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}$$

---

### ∂L/∂W⁽¹⁾ — Gradient w.r.t. hidden weights

$$\frac{\partial L}{\partial W^{(1)}} = \frac{\partial L}{\partial \text{pre-act}} \cdot x^T = \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix} \begin{bmatrix} -0.25 & -0.4 \end{bmatrix} = \begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix}$$

---

### ∂L/∂b⁽¹⁾ — Gradient w.r.t. hidden bias

$$\frac{\partial L}{\partial b^{(1)}} = \frac{\partial L}{\partial \text{pre-act}} = \mathbf{[0.0,\ 0.0]^T}$$

---

## Summary Table

| Gradient | Value |
|----------|-------|
| **L** | 0.853 |
| **∂L/∂ŷ** | 2.347 |
| **∂L/∂z** | 0.574 |
| **∂L/∂h** | [0.459, -0.344]ᵀ |
| **∂L/∂W⁽²⁾** | [0.0, 0.0] |
| **∂L/∂b⁽²⁾** | 0.574 |
| **∂L/∂W⁽¹⁾** | [[0,0],[0,0]] |
| **∂L/∂b⁽¹⁾** | [0.0, 0.0]ᵀ |

> ⚠️ **Dead ReLU Problem:** All gradients for W⁽¹⁾ and b⁽¹⁾ are **zero** because both hidden neurons received negative pre-activations, causing ReLU to zero them out. This means the first layer weights **receive no update** during this gradient step — a classic example of the *dying ReLU* problem, where neurons get permanently stuck and stop learning.
'''



## --------------------------- Answer for 2e
def text_to_average_embedding(text: str, vocab, embedding_layer: nn.Embedding) -> torch.Tensor:
    """
    Convert text to an averaged embedding vector using learnable embeddings.

    @param text: Input text string
    @param vocab: Vocabulary object
    @param embedding_layer: PyTorch embedding layer
    @return: A single tensor representing the averaged embedding
    """
    # Build list of vocabulary indices for each word in the text
    indices = [vocab[word] for word in text.split()]

    # Convert to a long tensor (required by nn.Embedding)
    indices_tensor = torch.tensor(indices, dtype=torch.long)

    # Look up the embedding for each word: shape (num_words, embedding_dim)
    embeddings = embedding_layer(indices_tensor)

    # Average across the word dimension → shape (embedding_dim,)
    averaged = reduce(embeddings, 'words embedding -> embedding', 'mean')

    return averaged


## --------------------------- Answer for 2d
'''
## Patterns in the 2D Embedding Space

First, let's organize the words by type and observe their coordinates:

| Category | Word | Embedding |
|----------|------|-----------|
| **Positive** | amazing | [**+0.8, +0.6**] |
| **Positive** | love | [**+0.9, +0.4**] |
| **Negative** | angry | [**-0.6, -0.8**] |
| **Negative** | hate | [**-0.8, -0.5**] |
| **Negative** | scared | [**-0.5, -0.7**] |
| **Negative** | worried | [**-0.3, -0.6**] |
| **Negative** | spiders | [-0.3, -0.4] |
| **Neutral** | of | [0.0, 0.0] |
| **Neutral** | this | [0.0, 0.0] |
| **Neutral** | so | [0.1, 0.0] |
| **Neutral** | about | [0.0, -0.1] |
| **Neutral** | day | [0.2, 0.1] |
| **Neutral** | tomorrow | [0.2, -0.2] |
| **Neutral** | waiting | [-0.1, -0.3] |

---

### i. Spatial Clustering

The embeddings reveal clear geometric clustering by sentiment: positive words ("amazing", "love") cluster in the upper-right quadrant with large positive values on both dimensions, while negative emotion words ("angry", "hate", "scared", "worried") cluster in the lower-left quadrant with consistently negative values. Neutral and function words ("of", "this", "so") cluster near the origin [0, 0], reflecting their lack of emotional content.

---

### ii. Semantic Similarity and Advantage over Bag-of-Words

Word embeddings encode semantic similarity as geometric proximity — words with similar meaning or emotional valence are placed close together in vector space, meaning operations like distance and direction carry real linguistic meaning. Unlike bag-of-words, which treats every word as an independent, orthogonal dimension (so "love" and "amazing" share no information), embeddings allow the model to generalize across synonyms and related words — a model that learns "hate" is negative automatically gains partial knowledge about "angry" simply because they occupy similar regions of the embedding space.
'''