import torch
import torchvision.transforms as transforms
from datagens.datagen import Datagen
from datagens.datasets.coco import CocoBbox


class CocoBboxDatagen(Datagen):
    def __init__(self, cfg):
        cfg = cfg["dataset"]
        batch_size = int(cfg["parameters"]["batch_size"])

        transform = transforms.Compose(
            [transforms.Resize([426, 640]),
             transforms.ToTensor()])

        train_dataset = CocoBbox(
            root=cfg["train"]["root"],
            annFile=cfg["train"]["ann"],
            transform=transform)
            
        test_dataset = CocoBbox(
            root=cfg["test"]["root"],
            annFile=cfg["test"]["ann"],
            transform=transform)

        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False)
