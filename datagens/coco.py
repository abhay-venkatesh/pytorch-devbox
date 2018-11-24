from datagens.datagen import Datagen
import torch
import torchvision
import torchvision.transforms as transforms


class CocoDatagen(Datagen):
    def __init__(self):
        train_dataset = torchvision.datasets.CocoDetection(
            root='../../data/coco/train2017',
            annFile='../../data/coco/annotations/instances_train2017.json',
            transform=transforms.ToTensor())

        test_dataset = torchvision.datasets.CocoDetection(
            root='../../data/coco/val2017',
            annFile='../../data/coco/annotations/instances_val2017.json',
            transform=transforms.ToTensor())

        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=5, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=5, shuffle=False)
