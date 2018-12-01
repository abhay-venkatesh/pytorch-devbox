from datagens.coco import CocoDatagen
from models.segnet import SegNet
import sys
from trainers.coco_trainer import Trainer
import yaml

if len(sys.argv) < 2:
    print("Usage: python main.py <path to config file>")

config_path = sys.argv[1]

with open(config_path, 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

datagen = CocoDatagen(cfg["coco"])
model = SegNet(1, 3)
trainer = Trainer(datagen, model)
trainer.run()
