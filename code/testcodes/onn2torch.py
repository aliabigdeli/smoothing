import torch
import torch.nn as nn
import onnx
from onnx2pytorch import ConvertModel
from torchvision import datasets, transforms

# Define the PyTorch model class
class MNISTRelu(nn.Module):
    def __init__(self):
        super(MNISTRelu, self).__init__()
        self.Flatten_7 = nn.Flatten()
        self.Gemm_8 = nn.Linear(in_features=784, out_features=256, bias=True)
        self.Relu_9 = nn.ReLU()
        self.Gemm_10 = nn.Linear(in_features=256, out_features=256, bias=True)
        self.Relu_11 = nn.ReLU()
        self.Gemm_12 = nn.Linear(in_features=256, out_features=10, bias=True)
        
        # self.model = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(in_features=784, out_features=256, bias=True),
        #     nn.ReLU(),
        #     nn.Linear(in_features=256, out_features=256, bias=True),
        #     nn.ReLU(),
        #     nn.Linear(in_features=256, out_features=10, bias=True)
        # )

    def forward(self, x):
        x = self.Flatten_7(x)
        x = self.Gemm_8(x)
        x = self.Relu_9(x)
        x = self.Gemm_10(x)
        x = self.Relu_11(x)
        x = self.Gemm_12(x)
        return x
        # return self.model(x)

# Load the ONNX file
onnx_file_path = 'mnist-net_256x2.onnx'
onnx_model = onnx.load(onnx_file_path)

# Convert the ONNX model to a PyTorch model using onnx2pytorch
pytorch_model_from_onnx = ConvertModel(onnx_model)

print("ONNX model converted to PyTorch model.")
print(pytorch_model_from_onnx)

# Initialize the MNISTRelu model
mnist_model = MNISTRelu()
print(mnist_model)

# Copy the weights from the converted PyTorch model to the MNISTRelu model
state_dict = pytorch_model_from_onnx.state_dict()
mnist_model.load_state_dict(state_dict, strict=False)
# Save the PyTorch model to a file
torch.save(mnist_model.state_dict(), 'mnist_model.pth')

print("Weights successfully loaded from ONNX file to PyTorch model.")

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor()])

# Load the MNIST test dataset
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Get a sample from the test dataset
sample_idx = 20
sample_image, sample_label = test_dataset[sample_idx]

# Add a batch dimension and pass the sample through the model
# sample_image = sample_image.unsqueeze(0)
output = mnist_model(sample_image)

# Get the predicted label
_, predicted_label = torch.max(output, 1)

# Check if the prediction is correct
is_correct = predicted_label.item() == sample_label

print(f"Predicted label: {predicted_label.item()}, Actual label: {sample_label}, Correct: {is_correct}")

# Create a DataLoader for the test dataset
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Function to calculate accuracy
def calculate_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

# Calculate and print the accuracy on the test dataset
accuracy = calculate_accuracy(mnist_model, test_loader)
print(f'Accuracy on the test dataset: {accuracy * 100:.2f}%')