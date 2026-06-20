import torch

x = torch.tensor(2.0, requires_grad=True)

y = x ** 2

y.backward()

print("x =", x)
print("y =", y)
print("dy/dx =", x.grad)