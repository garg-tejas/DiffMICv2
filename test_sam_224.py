import torch
from model import SamEncoder

print("Testing SamEncoder at 224x224...")
m = SamEncoder(image_size=224)
x = torch.zeros(2, 3, 224, 224)
with torch.no_grad():
    out = m(x)
print("Output shape:", out.shape)
print("Success! No OOM.")
