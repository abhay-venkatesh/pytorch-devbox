import torch
import torchvision.transforms as transforms
from datagens.datagen import Datagen
from datasets.datasets.coco import CocoStuff


class CocoDatagen(Datagen):
    def __init__(self, cfg):
        cfg = cfg["dataset"]

        transform = transforms.Compose(
            [transforms.Resize([426, 640]),
             transforms.ToTensor()])

        train_dataset = CocoStuff(
            root=cfg["train"]["root"],
            annFile=cfg["train"]["ann"],
            transform=transform)
            
        self.total_steps = train_dataset.total_steps

        test_dataset = CocoStuff(
            root=cfg["test"]["root"],
            annFile=cfg["test"]["ann"],
            transform=transform)

        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=1, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=1, shuffle=False)
