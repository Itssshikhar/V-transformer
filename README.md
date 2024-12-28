# Vision Transformer (ViT) from Scratch

This repository hosts the implementation of a Vision Transformer (ViT) model trained on the CIFAR-10 dataset. Vision Transformers have revolutionized computer vision by adapting transformer models, traditionally used in NLP, for image-related tasks.

## Repository Link

Access the complete project on GitHub: [V-transformer](https://github.com/Itssshikhar/V-transformer)

---

## Project Overview

### Objective
The goal of this project is to explore the application of Vision Transformers on the CIFAR-10 dataset and evaluate their performance compared to traditional convolutional neural networks (CNNs).

### Key Features
- Implementation of a Vision Transformer model from scratch.
- Training and evaluation pipelines designed for CIFAR-10.
- Utilization of PyTorch for model development and training.
- Performance comparison metrics and visualization tools.

---

## Getting Started

### Prerequisites
Ensure you have the following installed:

- Python >= 3.8
- PyTorch >= 1.12.0
- torchvision
- numpy
- matplotlib

Install the required libraries:

```bash
pip install -r requirements.txt
```

### Dataset
The CIFAR-10 dataset is automatically downloaded using `torchvision.datasets`.

---

## Model Architecture

The Vision Transformer model is built as follows:

1. **Patch Embedding**: The input image is divided into smaller patches, flattened, and linearly projected.
2. **Transformer Encoder**: Stacked transformer layers equipped with multi-head self-attention and feedforward layers.
3. **Classification Head**: A linear layer maps the output of the encoder to the 10 class labels.

### Code Example

```python
import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, num_classes, dim, depth, heads, mlp_dim):
        super(VisionTransformer, self).__init__()
        # Patch embedding layer
        self.patch_embed = nn.Linear(patch_size**2 * 3, dim)
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, (img_size // patch_size)**2 + 1, dim))
        # Transformer layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim),
            num_layers=depth
        )
        # Classification head
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, x, output_attentions=False):
      #Calculate the embedding output
      embedding_output = self.embedding(x)
      #Calculate the encoder's output
      encoder_output, all_attentions = self.encoder(embedding_output, output_attentions=output_attentions)
      #Calculate the logits, taking the Classify token's output as feature for classfication
      logits = self.classifier(encoder_output[:, 0])
      #Return the logits and the attention probabailities
      if not output_attentions:
         return(logits, None)
      else:
         return(logits, all_attentions)
```

---

## Results

### Training Metrics
- **Accuracy**: The model achieved an accuracy of ~90% on the test set.
- **Loss**: Training and validation loss decreased consistently over epochs.

### Visualizations
![Training Accuracy Curve](path/to/accuracy_curve.png)

### Comparisons
The ViT outperformed baseline CNN models for CIFAR-10 in terms of accuracy, demonstrating the effectiveness of transformer-based architectures for vision tasks.

---

## Usage

### Training
To train the model, run:

```bash
python train.py
```

### Evaluation
Evaluate the trained model:

```bash
python evaluate.py --checkpoint <path_to_checkpoint>
```

---

## References

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2. [An Image Is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)

---

Feel free to contribute or raise issues on the [GitHub repository](https://github.com/Itssshikhar/V-transformer).
