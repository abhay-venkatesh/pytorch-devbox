import torch
import torchvision.transforms as transforms
from datagens.datagen import Datagen
from datasets.coco import CocoStuff


class CocoDatagen(Datagen):
    def __init__(self):
        transform = transforms.Compose(
            [transforms.Resize([426, 640]),
             transforms.ToTensor()])

        train_dataset = CocoStuff(
            root='../../data/coco/train2017',
            annFile='../../data/coco/annotations/stuff_train2017.json',
            transform=transform)

        test_dataset = CocoStuff(
            root='../../data/coco/val2017',
            annFile='../../data/coco/annotations/stuff_val2017.json',
            transform=transform)

        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=1, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=1, shuffle=False)