# Week3 U-Net

## Dataset

ISBI 2012 EM Segmentation

Training Images : 30

Validation : 6

Input Size

```
256×256
```

---

## Model

Original U-Net

Encoder

```
64
128
256
512
1024
```

Decoder

```
512
256
128
64
```

---

## Loss

Experiment 1

```
BCEWithLogitsLoss
```

Experiment 2

```
BCEWithLogitsLoss
+
Dice Loss
```

---

## Optimizer

```
Adam
```

LR

```
1e-3
```

Epoch

```
30
```

Batch

```
2
```

---

## Result

Baseline

Dice

```
0.9515
```

IoU

```
0.9075
```

---

BCE + Dice

Dice

```
0.9520
```

IoU

```
0.9085
```

---

Conclusion

```
Dice Loss가 Validation Dice와 IoU를
소폭 향상시켰으며,
Validation Loss는 더욱 안정적으로 감소하였다.
```

## Files

dataset.py
- ISBI2012 Dataset 구현

unet.py
- Original U-Net 모델 구현

train.py
- BCE Loss 학습

train_bce_dice.py
- BCE + Dice Loss 학습

test.py
- 검증 및 예측 이미지 저장

plot_history.py
- 학습 결과 비교 그래프 생성

## Author

CVIP Tutorial Week3
Biomedical Image Segmentation using U-Net