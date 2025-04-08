import torch

import torchvision.models as models

# Load the ResNet-50 model
resnet50 = models.resnet50(pretrained=True)

# Print the model architecture
print(resnet50)

# Set the model to evaluation mode
resnet50.eval()

# Create dummy input with the same size as the model's input
dummy_input = torch.randn(1, 3, 224, 224)

# Export the model to ONNX format
torch.onnx.export(resnet50, dummy_input, "resnet50.onnx", 
                  input_names=['input'], output_names=['output'], 
                  opset_version=11)