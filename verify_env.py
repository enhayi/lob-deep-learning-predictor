import torch
import sys

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Device Count:", torch.cuda.device_count())
    print("Current Device:", torch.cuda.get_device_name(0))
    print("CUDA Version:", torch.version.cuda)
    x = torch.randn(1000, 1000, device="cuda")
    y = x @ x
    print("Matrix multiplication on GPU succeeded. Result sum:", y.sum().item())
else:
    print("WARNING: CUDA is not available on this device!")
