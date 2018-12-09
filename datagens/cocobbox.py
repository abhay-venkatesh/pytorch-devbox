import torch
import torchvision.transforms as transforms
from datagens.datagen import Datagen
from datagens.datasets.coco import CocoBbox


class CocoBboxDatagen(Datagen):
    def __init__(self, cfg):
        batch_size = int(cfg["parameters"]["batch_size"])
        cfg = cfg["dataset"]

        train_dataset = CocoBbox(
            root=cfg["train"]["root"],
            ann_file_path=cfg["train"]["ann"])
            
        test_dataset = CocoBbox(
            root=cfg["test"]["root"],
            ann_file_path=cfg["test"]["ann"])
            
        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False)
