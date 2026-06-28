import torch

print(f"PyTorch version: {torch.__version__}")
print(f"Is CUDA available (ROCm via PyTorch)? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
else:
    print("CUDA (ROCm) is not available, PyTorch will use CPU.")

# Test simple de calcul sur le CPU
try:
    x = torch.rand(5, 3)
    print(f"Tensor on CPU: {x.device}")
except Exception as e:
    print(f"Failed to run a tensor on CPU: {e}")

