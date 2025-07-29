"""Benchmarks a simple PyTorch model on macOS using the Metal Performance Shaders (MPS) backend."""

import time

import torch


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

model = torch.nn.Linear(1000, 1000).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

x = torch.randn(64, 1000).to(device)
y = torch.randn(64, 1000).to(device)

ITERS = 1000
start = time.time()
for i in range(ITERS):
    optimizer.zero_grad()
    output = model(x)
    loss = torch.nn.functional.mse_loss(output, y)
    loss.backward()
    optimizer.step()
end = time.time()

print(f"Avg time per iter: {(end - start) / ITERS:.6f} sec")
