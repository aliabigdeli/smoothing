import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from architectures import get_architecture
from datasets import get_dataset

# Import the Smooth class from the repository.
# (Adjust the import path below according to your repository structure.)
from core import Smooth
import csv

def get_model():
    """
    Constructs a ResNet18 model modified for CIFAR-10.
    Replace or update this function as needed to load your trained classifier.
    """

    # load the base classifier
    # checkpoint = torch.load(args.base_classifier)
    checkpoint = torch.load('models/cifar10/resnet110/noise_0.25/checkpoint.pth.tar')
    # base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    base_classifier = get_architecture(checkpoint["arch"], "cifar10")
    base_classifier.load_state_dict(checkpoint['state_dict'])
    return base_classifier


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load a single image from CIFAR-10 (here, we take the first image from the test set)
    # transform = transforms.Compose([transforms.ToTensor()])
    # cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # img, label = cifar10_test[0]  # img shape: (3, 32, 32)
    
    
    # dataset = get_dataset(args.dataset, args.split)
    dataset = get_dataset("cifar10", "test")
    (img, label) = dataset[0]
    img = img.unsqueeze(0).to(device)  # add batch dimension: (1, 3, 32, 32)


    # Load your base classifier.
    model = get_model()
    model.to(device)
    model.eval()

    # Set up the smoothed classifier.
    sigma = 0.25  # noise level; adjust as needed
    num_classes = 10
    smoothed_classifier = Smooth(model, num_classes, sigma)

    # Certification parameters
    N0 = 100     # number of samples for initial prediction
    N = 10000    # number of samples for certification
    alpha = 0.001  # failure probability

    # Certify the prediction: returns (predicted_class, certified_radius)
    pred_class, certified_radius = smoothed_classifier.certify(img, N0, N, alpha, batch_size=100)
    print(f"Label lass: {label}")
    print("Predicted class:", pred_class)
    print("Certified radius (L2):", certified_radius)

    n_noise_imgs = 1

    if n_noise_imgs == 1 :

        ### mode 1: equal perturbation to all pixels
        # Add a perturbation with 90% of the certified L2 radius.
        # We sample a random noise vector and then scale it to have the desired L2 norm.
        noise = torch.randn_like(img, device=device)
        noise_flat = noise.view(noise.size(0), -1)  # flatten for norm computation
        noise_norm = noise_flat.norm(p=2, dim=1, keepdim=True)  # shape: (1, 1)
        desired_norm = 1.1 * certified_radius  # scalar: 90% of the certified radius

        # Scale the noise so that its L2 norm equals desired_norm.
        scaling_factor = desired_norm / noise_norm.item()
        perturbation = noise * scaling_factor

        # Create the perturbed image and ensure it stays in valid range [0,1].
        perturbed_img = img + perturbation
        perturbed_img = torch.clamp(perturbed_img, 0.0, 1.0)

        # (Optional) Verify the norm of the perturbation.
        perturbation_norm = torch.norm(perturbation.view(perturbation.size(0), -1), p=2, dim=1).item()
        print("Perturbation L2 norm:", perturbation_norm)

        # (Optional) Check the prediction on the perturbed image.
        pred_class_perturbed, _ = smoothed_classifier.certify(perturbed_img, N0, N, alpha, batch_size=100)
        print("Predicted class on perturbed image:", pred_class_perturbed)

        ### mode 2: perturbation to only one pixel
        print("mode 2: perturbation to only one pixel")
        x_max, y_max = img.shape[-2], img.shape[-1]
        x_idx, y_idx = np.random.randint(0, x_max), np.random.randint(0, y_max)
        for fac in [5.0, 10.0, 20.0, 50.0]:
            for i in range(x_max):
                for j in range(y_max):
                    noise = torch.zeros_like(img, device=device)
                    noise[0, :, i, j] = 1.0
                    # noise[0, :, x_idx, y_idx] = 1.0
                    # print(noise[0, :, x_idx, y_idx])
                    # print(f"noise.shape: {noise.shape}")
                    noise_flat = noise.view(noise.size(0), -1)  # flatten for norm computation
                    noise_norm = noise_flat.norm(p=2, dim=1, keepdim=True)  # shape: (1, 1)
                    desired_norm = fac * certified_radius  # scalar: 90% of the certified radius

                    # Scale the noise so that its L2 norm equals desired_norm.
                    scaling_factor = desired_norm / noise_norm.item()
                    perturbation = noise * scaling_factor

                    # Create the perturbed image and ensure it stays in valid range [0,1].
                    perturbed_img = img + perturbation
                    perturbed_img = torch.clamp(perturbed_img, 0.0, 1.0)

                    # (Optional) Verify the norm of the perturbation.
                    perturbation_norm = torch.norm(perturbation.view(perturbation.size(0), -1), p=2, dim=1).item()
                    # print("Perturbation L2 norm:", perturbation_norm)

                    # (Optional) Check the prediction on the perturbed image.
                    pred_class_perturbed, _ = smoothed_classifier.certify(perturbed_img, N0, N, alpha, batch_size=100)
                    # print("Predicted class on perturbed image:", pred_class_perturbed)
                    if pred_class_perturbed != pred_class and pred_class_perturbed != -1:
                        with open("code/changed_pixels.csv", "a", newline="") as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow([fac, i, j])
                        print(f"### Prediction changed at pixel ({i}, {j})")

    elif n_noise_imgs > 1:
        # Create a batch of noise images with correct dimensions.
        noise = torch.randn(n_noise_imgs, *img.shape[1:], device=device)  # fixed shape
        noise_flat = noise.view(n_noise_imgs, -1)  # flatten each image
        noise_norm = noise_flat.norm(p=2, dim=1, keepdim=True)  # shape: (n_noise_imgs, 1)
        desired_norm = 1.0 * certified_radius
        scaling_factor = desired_norm / noise_norm  # shape: (n_noise_imgs, 1)
        perturbation = noise * scaling_factor.view(n_noise_imgs, 1, 1, 1)  # broadcast correctly
        perturbed_imgs = img.expand(n_noise_imgs, -1, -1, -1) + perturbation  # expand img for batch addition
        
        # Iterate over each perturbed image to certify individually.
        pred_classes_perturbed = []
        for i in range(n_noise_imgs):
            pred, _ = smoothed_classifier.certify(perturbed_imgs[i:i+1], N0, N, alpha, batch_size=100)
            pred_classes_perturbed.append(pred)
        pred_classes_perturbed = torch.tensor(pred_classes_perturbed)
        
        print("Predicted classes on perturbed images:", pred_classes_perturbed)
        n_correct = (pred_classes_perturbed == pred_class).sum().item()
        print(f"Number of correct predictions: {n_correct} / {n_noise_imgs}")
        abstained = (pred_classes_perturbed == -1).sum().item()
        print(f"Number of abstained predictions: {abstained} / {n_noise_imgs}")
        uncorrect = n_noise_imgs - n_correct - abstained
        print(f"Number of incorrect predictions: {uncorrect} / {n_noise_imgs}")
        

        
        

if __name__ == '__main__':
    main()
