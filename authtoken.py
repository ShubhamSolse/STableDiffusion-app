auth_token = ""


import torch

print("PyTorch version:", torch.__version__)
print("CUDA version used by PyTorch binaries:", torch.version.cuda)
print("Is CUDA available?", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device")

