# CVIP Tutorial

Gachon University CVIP Lab  
Stage 1 Tutorial

## Progress

### Week0
- ✓ Python

### Week1
- ✓ CNN

### Week2
- ✓ ResNet18
- ✓ Transfer Learning

### Week3
- ✓ U-Net
- ✓ Semantic Segmentation
- ✓ BCE
- ✓ BCE + Dice Loss

### Week4
- ✓ PSPNet
- ✓ Pyramid Pooling Module
- ✓ Semantic Segmentation

### Week5
- ✓ FPN
- ✓ Multi-scale Feature
- ✓ Lateral Connection

### Week6 - Vision Transformer (ViT)
- ✓ Vision Transformer implementation
- ✓ Patch Embedding
- ✓ Multi-Head Self-Attention
- ✓ Transformer Encoder
- ✓ Image Classification
- ✓ Ablation Study

### Week7 - ConvNeXt
- ✓ ConvNeXt architecture implementation
- ✓ Depthwise Convolution
- ✓ 1×1 Pointwise Convolution
- ✓ Inverted Bottleneck
- ✓ Layer Normalization
- ✓ GELU
- ✓ Patchify Stem
- ✓ Large Kernel Convolution
- ✓ 7×7 vs 3×3 kernel ablation
- ✓ Accuracy / Precision / Recall / F1 analysis
- ✓ Confusion Matrix analysis
- ✓ Class-wise F1 analysis
- ✓ Training curve comparison

#### ConvNeXt Result

| Model | Test Accuracy | Macro F1 |
|---|---:|---:|
| ConvNeXt 7×7 | 72.60% | 72.00% |
| ConvNeXt 3×3 | **75.77%** | **75.01%** |

Detailed experiments:  
[`week7_convnext/README.md`](week7_convnext/README.md)

---

## Repository Structure

```text
cvip-tutorial/
├── week0/
├── week1/
├── week2/
├── week3_unet/
├── week4_pspnet/
├── week5_fpn/
├── week6_vit/
├── week7_convnext/
└── reports/
````

## Environment

* Python 3.12
* PyTorch
* Torchvision
* OpenCV
* Matplotlib
* scikit-learn
* pandas

## Reports

### Semantic Segmentation

* U-Net / PSPNet / FPN comparison
* `reports/segmentation_comparison/model_comparison.md`

### Image Classification

* ViT experiments: [`week6_vit/README.md`](week6_vit/README.md)
* ConvNeXt experiments: [`week7_convnext/README.md`](week7_convnext/README.md)

## Author

김수민