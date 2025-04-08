import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np

# Import modules from the smoothing-adversarial repo.
# Adjust these imports according to the repository structure.
# from smoothing import Smoothing  # This should wrap a base classifier with randomized smoothing.
# from torchvision.models.resnet import ResNet110  # Assumed location of the ResNet110 definition.
from core import Smooth  # This is the class for the smoothed classifier.


from architectures import get_architecture
from datasets import get_dataset

def get_model(checkpoint_path='../models/cifar10/resnet110/noise_0.25/checkpoint.pth.tar', dataset='cifar10'):
    """
    load the base classifier from the checkpoint
    :param checkpoint_path: path to the checkpoint
    :param dataset: dataset name
    :return: the base classifier
    """

    # load the base classifier
    checkpoint = torch.load(checkpoint_path)
    base_classifier = get_architecture(checkpoint["arch"], dataset)
    base_classifier.load_state_dict(checkpoint['state_dict'])
    return base_classifier

# Function to estimate the gradient via EOT.
def estimate_gradient(model, x, y, num_samples, sigma):
    """
    Estimates the gradient of the loss w.r.t x by averaging gradients
    computed on noisy copies of x.
    """
    grad = torch.zeros_like(x)
    for _ in range(num_samples):
        noise = torch.randn_like(x) * sigma
        # Create a noisy version of the input.
        x_noisy = (x + noise).detach()
        x_noisy.requires_grad = True

        # Compute the loss on the base classifier.
        output = model(x_noisy)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        grad += x_noisy.grad.detach()
    return grad / num_samples

# PGD attack that uses EOT to compute gradients.
def pgd_attack(smoothed_classifier, x, y, epsilon, alpha, num_iter, num_samples, sigma, verbose=False):
    """
    Runs a PGD attack on the smoothed classifier.
    smoothed_classifier: an instance of Smoothing wrapping the base classifier.
    x: original input image.
    y: true label.
    epsilon: maximum perturbation allowed.
    alpha: step size.
    num_iter: number of PGD iterations.
    num_samples: number of noise samples per iteration for gradient estimation.
    sigma: noise level used for smoothing.
    """
    # Initialize adversarial example.
    x_adv = x.clone().detach().to(x.device)
    x_adv.requires_grad = True
    
    for i in range(num_iter):
        # Estimate the gradient on the base classifier using EOT.
        grad = estimate_gradient(smoothed_classifier.base_classifier, x_adv, y, num_samples, sigma)
        
        # Perform the PGD update: take a step in the direction of the sign of the gradient.
        x_adv = x_adv + alpha * torch.sign(grad)
        
        # Project the perturbation onto the epsilon ball around the original input.
        x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)
        
        # Ensure the adversarial image is valid (e.g. in [0,1] range).
        x_adv = torch.clamp(x_adv, 0, 1).detach().clone()
        x_adv.requires_grad_()
        
        if i % 10 == 0 and verbose:
            print(f"L_inf PGD attack, Iteration {i} completed.")
    return x_adv.detach()

def pgd_attack_l2(smoothed_classifier, x, y, epsilon, alpha, num_iter, num_samples, sigma, verbose=False):
    """
    Performs a PGD attack in the L2 norm ball on the smoothed classifier using EOT.
    
    smoothed_classifier: an instance of Smoothing wrapping the base classifier.
    x: original input image.
    y: true label.
    epsilon: L2 norm radius for the perturbation.
    alpha: step size per iteration.
    num_iter: total number of PGD iterations.
    num_samples: number of noise samples for gradient estimation.
    sigma: noise level used for smoothing.
    """
    x_adv = x.clone().detach().to(x.device)
    x_adv.requires_grad = True

    for i in range(num_iter):
        # Estimate the gradient using EOT.
        grad = estimate_gradient(smoothed_classifier.base_classifier, x_adv, y, num_samples, sigma)
        
        # Normalize the gradient in L2 norm.
        grad_norm = torch.norm(grad.view(grad.shape[0], -1), p=2, dim=1).view(-1, 1, 1, 1)
        grad_norm = torch.clamp(grad_norm, min=1e-8)
        normalized_grad = grad / grad_norm
        
        # Update: take a step in the direction of the normalized gradient.
        x_adv = x_adv + alpha * normalized_grad

        # Project the perturbation onto the L2 ball of radius epsilon.
        delta = x_adv - x
        delta_norm = torch.norm(delta.view(delta.shape[0], -1), p=2, dim=1).view(-1, 1, 1, 1)
        # If delta_norm > epsilon, scale it back to the ball.
        delta = delta * (epsilon / (delta_norm + 1e-8)).clamp(max=1.0)
        x_adv = x + delta

        # Ensure the adversarial example is in a valid range.
        x_adv = torch.clamp(x_adv, 0, 1).detach().clone()
        x_adv.requires_grad_()

        if i % 10 == 0 and verbose:
            print(f"L2 PGD attack, Iteration {i} completed.")

    return x_adv.detach()

def main():
    # dataset_name = 'cifar10'
    # dataset = get_dataset(dataset_name, "test")

    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define the transformation for CIFAR-10.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load CIFAR-10 test set.
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)
    
    # # Load the pretrained ResNet110 base classifier.
    # model = ResNet110(num_classes=10).to(device)
    # checkpoint = torch.load('path_to_resnet110_cifar10_checkpoint.pth', map_location=device)
    # model.load_state_dict(checkpoint['state_dict'])
    # model.eval()

    # Load your base classifier.
    base_model = get_model()
    base_model.to(device)
    base_model.eval()
    
    # Wrap the base classifier with the smoothing module using sigma = 0.25.
    sigma = 0.25
    # smoothed_classifier = Smoothing(model, num_classes=10, sigma=sigma)
    smoothed_classifier = Smooth(base_model, num_classes=10, sigma=sigma)
    
    # Set attack parameters.
    epsilon = 0.05   # maximum allowed perturbation
    alpha = 0.01     # step size per iteration
    num_iter = 40    # total PGD iterations
    num_samples = 50 # number of noise samples per iteration for gradient estimation
    
    # Run the attack on one example from the test set.
    for data, target in testloader:
        data, target = data.to(device), target.to(device)
        
        # Obtain the original prediction using a high number of samples.
        orig_pred = smoothed_classifier.predict(data, n=1000, alpha=alpha, batch_size=100)
        print("Original prediction:", orig_pred)
        
        # Craft an adversarial example using our PGD with EOT attack.
        adv_data = pgd_attack(smoothed_classifier, data, target, epsilon, alpha, num_iter, num_samples, sigma)
        
        # Get the prediction for the adversarial example.
        adv_pred = smoothed_classifier.predict(adv_data, 1000, alpha, 100)
        print("L_inf Adversarial prediction:", adv_pred)
        break  # Remove or adjust this break to attack more examples.

    # Attack parameters.
    epsilon = 0.5   # L2 norm budget (adjust based on desired strength)
    alpha = 0.1     # step size
    num_iter = 40   # number of PGD iterations
    num_samples = 50  # number of noise samples per iteration for gradient estimation
    
    # Run the attack on one example from the test set.
    for data, target in testloader:
        data, target = data.to(device), target.to(device)
        
        # Obtain the original prediction using a high number of samples.
        orig_pred = smoothed_classifier.predict(data, n=1000, alpha=alpha, batch_size=100)
        print("Original prediction:", orig_pred)
        
        # Craft an adversarial example using the L2 PGD attack.
        adv_data = pgd_attack_l2(smoothed_classifier, data, target, epsilon, alpha, num_iter, num_samples, sigma)
        
        # Get the prediction for the adversarial example.
        adv_pred = smoothed_classifier.predict(adv_data, 1000, alpha, 100)
        print("L_2 Adversarial prediction:", adv_pred)
        break  # Remove or adjust this break to attack more examples.

if __name__ == '__main__':
    main()
