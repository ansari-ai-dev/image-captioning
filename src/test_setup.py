import torch, torchvision
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

from torchvision.models import resnet50, ResNet50_Weights
model = resnet50(weights=ResNet50_Weights.DEFAULT)
x = torch.randn(1, 3, 224, 224)
out = model(x)
print("ResNet50 output shape:", out.shape)