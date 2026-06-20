import torch

print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

x = torch.tensor([1, 2, 3])

print(x)
print(x.shape)