# Semantic Segmentation Model Comparison

| Model | Parameters | Best Epoch | Validation Dice | Validation IoU |
|---|---:|---:|---:|---:|
| U-Net + BCE/Dice | 7,762,465 | 20 | 0.9520 | 0.9085 |
| FPN | 5,870,433 | 23 | 0.9497 | 0.9043 |
| Lightweight PSPNet | 1,708,705 | 29 | 0.8906 | 0.8030 |

## Conclusion

U-Net은 skip connection을 통해 고해상도 공간 정보를 직접 전달하여
세밀한 세포막 경계 복원에서 가장 높은 성능을 기록하였다.

FPN은 multi-scale feature와 lateral connection을 활용하여
U-Net보다 적은 파라미터로 유사한 성능을 기록하였고,
성능과 모델 크기 사이에서 가장 균형 잡힌 결과를 보였다.

Lightweight PSPNet은 Pyramid Pooling Module을 통해
넓은 문맥 정보를 활용할 수 있었지만,
저해상도 feature를 직접 업샘플링하는 구조로 인해
얇은 경계의 누락이 많아 가장 낮은 성능을 보였다.