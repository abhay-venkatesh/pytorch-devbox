from datagens.datagen import Datagen
import torch
import torchvision
import torchvision.transforms as transforms


class CifarDatagen(Datagen):
    def __init__(self):
        # Image preprocessing modules
        transform = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor()])

        train_dataset = torchvision.datasets.CIFAR10(
            root='../../data/', train=True, transform=transform, download=True)

        test_dataset = torchvision.datasets.CIFAR10(
            root='../../data/', train=False, transform=transforms.ToTensor())

        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=100, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=100, shuffle=False)
