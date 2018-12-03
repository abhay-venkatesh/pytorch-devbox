from datagens.coco import CocoDatagen
from models.segnet import SegNet
import sys
from trainers.coco_trainer import Trainer
import yaml

if len(sys.argv) < 2:
    print("Usage: python main.py <path to config file>")

config_path = sys.argv[1]
with open(config_path, 'r') as ymlfile:
    config = yaml.load(ymlfile)
    
checkpoint_path = None
if len(sys.argv) > 2:
    checkpoint_path = sys.argv[2]

datagen = CocoDatagen(config)
model = SegNet(1, 3)
trainer = Trainer(datagen, model, config)
trainer.run(checkpoint_path)
