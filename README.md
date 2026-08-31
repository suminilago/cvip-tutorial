# CVIP Tutorial

Gachon University CVIP Lab  
Stage 1 Tutorial

---

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
- ✓ Label Smoothing experiment
- ✓ Strong Augmentation experiment
- ✓ Longer Training experiment
- ✓ LayerScale + DropPath experiment
- ✓ Pretrained ConvNeXt-Tiny Transfer Learning
- ✓ Test Accuracy 80%+ target achieved

#### ConvNeXt Initial Experiments

| Model | Test Accuracy | Macro F1 |
|---|---:|---:|
| Custom ConvNeXt 7×7 | 72.60% | 72.00% |
| Custom ConvNeXt 3×3 | **75.77%** | **75.01%** |

Kernel size를 7×7에서 3×3으로 변경한 결과,
Test Accuracy가 **72.60% → 75.77%**로 **3.17%p 향상**되었다.

#### ConvNeXt Performance Improvement

3×3 Custom ConvNeXt를 기준으로 Test Accuracy 80% 이상을 목표로
추가적인 성능 개선 실험을 수행하였다.

| Experiment | Best Val Acc | Test Acc | Macro F1 |
|---|---:|---:|---:|
| Custom ConvNeXt 3×3 | 78.47% | 75.77% | 75.01% |
| + Label Smoothing | 79.33% | 76.60% | 75.39% |
| + Strong Augmentation | 77.70% | - | - |
| + Longer Training | **80.37%** | **78.23%** | **76.91%** |
| + LayerScale + DropPath | 75.60% | - | - |
| **Pretrained ConvNeXt-Tiny** | **87.43%** | **88.38%** | **88.23%** |

Custom ConvNeXt에서는 Label Smoothing과 Longer Training을 통해
Test Accuracy를 **75.77% → 78.23%**까지 향상시켰다.

이후 ImageNet pretrained ConvNeXt-Tiny를 이용한 Transfer Learning을 적용하여
최종 Test Accuracy **88.38%**를 달성하였다.

> **Final ConvNeXt Result: Test Accuracy 88.38%**

Detailed reports:

- [Week7 ConvNeXt Implementation & Ablation](week7_convnext/README.md)
- [Week7 ConvNeXt Performance Improvement Report](week7_convnext/IMPROVEMENT_REPORT.md)

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
│   ├── README.md
│   ├── IMPROVEMENT_REPORT.md
│   ├── dataset.py
│   ├── model.py
│   ├── model_improved.py
│   ├── train.py
│   ├── train_kernel3.py
│   ├── train_improved_ls.py
│   ├── train_improved_aug.py
│   ├── train_improved_long.py
│   ├── train_improved_model.py
│   ├── train_transfer.py
│   ├── test.py
│   ├── test_kernel3.py
│   ├── test_improved_ls.py
│   ├── test_improved_long.py
│   ├── test_transfer.py
│   └── result/
└── reports/
```

---

## Environment

- Python 3.12
- PyTorch
- Torchvision
- OpenCV
- Matplotlib
- scikit-learn
- pandas

---

## Reports

### Semantic Segmentation

- U-Net / PSPNet / FPN comparison
- `reports/segmentation_comparison/model_comparison.md`

### Image Classification

- ViT experiments: [`week6_vit/README.md`](week6_vit/README.md)
- ConvNeXt implementation & ablation: [`week7_convnext/README.md`](week7_convnext/README.md)
- ConvNeXt performance improvement: [`week7_convnext/IMPROVEMENT_REPORT.md`](week7_convnext/IMPROVEMENT_REPORT.md)

---

## Author

김수민