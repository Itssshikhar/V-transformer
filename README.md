# Vision Transformer (ViT) from Scratch

## Project Description
This project implements a Vision Transformer (ViT) from scratch using PyTorch. The Vision Transformer is a model for image classification that leverages the Transformer architecture, originally designed for natural language processing, and applies it to image patches. This approach allows the model to capture long-range dependencies and global context in images more effectively than traditional convolutional neural networks (CNNs).

## Setup

### Requirements
- Python 3.x
- PyTorch
- torchvision
- DeepSpeed (optional, for memory optimization)
- NVIDIA GPU with CUDA support (optional but recommended)

### Installation
Install the required packages using pip:
```bash
pip install torch torchvision deepspeed
```

### Configuration
Configure the ViT model parameters in a dictionary format:
```
config = {
    "patch_size": 4,
    "hidden_size": 48,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 4 * 48,
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "image_size": 32,
    "num_classes": 10,
    "num_channels": 3,
    "qkv_bias": True,
    "use_faster_attention": True,
}

```
