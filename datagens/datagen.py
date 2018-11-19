import torchvision
import torchvision.transforms as transforms


class DataGen:
    def __init__(self):
        train_dataset = torchvision.datasets.MNIST(
            root='../../data',
            train=True,
            transform=transforms.ToTensor(),
            download=True)

        test_dataset = torchvision.datasets.MNIST(
            root='../../data', train=False, transform=transforms.ToTensor())

        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False)
