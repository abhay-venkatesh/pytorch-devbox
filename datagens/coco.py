from datagens.datagen import Datagen
import torch
import torchvision
import torchvision.transforms as transforms


class CocoDatagen(Datagen):
    def __init__(self):
        train_dataset = torchvision.datasets.CocoDetection(
            root='../../data/',
            transform=transforms.ToTensor(),
            download=True)

        test_dataset = torchvision.datasets.CocoDetection(
            root='../../data/', transform=transforms.ToTensor())

        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=100, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=100, shuffle=False)
