# Week0

## 1. 환경 구축
- PyTorch 설치

## 2. 실습

## 실험 결과

### Tensor

- tensor([1,2,3]) 생성
- shape 확인
- mean 계산

결과

tensor([5., 7., 9.])
tensor([4., 10., 18.])
tensor(2.)

### Autograd

- y=x^2
- x = 2일 때 dy/dx = 4 계산

결과

x = tensor(2., requires_grad=True)
y = tensor(4., grad_fn=<PowBackward0>)
dy/dx = tensor(4.)

### MNIST

- Dataset
- DataLoader

### MNIST Loader

images.shape = [64,1,28,28]

labels.shape = [64]

### MLP

입력:

[64,1,28,28]

출력:

[64,10]

## 3. 결과 분석

- Tensor는 딥러닝에서 사용하는 기본 데이터 구조이다.
- mean()은 float Tensor에서 사용 가능하다.
- Autograd는 미분을 자동으로 계산해준다.
- requires_grad=True를 사용하면 자동 미분이 가능하다.
- DataLoader는 데이터를 batch 단위로 불러온다.
- MLP는 이미지를 분류할 수 있다.
- MLP는 이미지를 10개의 클래스로 분류하기 위한 출력 벡터를 생성한다.

## 4. 배운 점

- Pytorch 환경 구축 방법을 익혔다.
- Tensor와 Autograd의 기본 사용법을 익혔다.
- MNIST 데이터셋을 불러오는 방법을 익혔다.
- MLP 모델의 입력과 출력 구조를 이해했다.

## 향후 계획

- Loss Function 학습
- Optimizer 학습
- MNIST MLP Training 구현
- Accuracy 측정