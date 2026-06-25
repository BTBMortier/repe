import torch

print("--- Diagnostic Matériel ---")
print(f"Version de PyTorch : {torch.__version__}")

# Pour ROCm (détecté comme CUDA par PyTorch)
cuda_available = torch.cuda.is_available()
print(f"GPU AMD ROCm (détecté CUDA) disponible : {cuda_available}")
if cuda_available:
    print(f"Nom de la carte : {torch.cuda.get_device_name(0)}")
    print(f"Nombre de GPUs : {torch.cuda.device_count()}")
else:
    # Test DirectML pour Windows
    try:
        import torch_directml
        dml = torch_directml.device()
        print("✓ Accélération DirectML disponible via le package torch-directml.")
    except ImportError:
        print("✗ Pas d'accélération matérielle GPU détectée. Exécution par défaut en CPU.")

