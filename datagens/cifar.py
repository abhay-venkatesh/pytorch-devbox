from datagen import DataGen
import torchvision
import torchvision.transforms as transforms


class CifarDatagen(DataGen):
    def __init__(self):
        train_dataset = torchvision.datasets.CIFAR10(
            root='../../data/', train=True, transform=transform, download=True)

        test_dataset = torchvision.datasets.CIFAR10(
            root='../../data/', train=False, transform=transforms.ToTensor())

        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=100, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=100, shuffle=False)
