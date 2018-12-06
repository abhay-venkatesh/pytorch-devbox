from datagens.cocostuff import CocoStuffDatagen
from models.segnet import SegNet
from trainers.trainer import Trainer
import sys
import yaml

if len(sys.argv) < 2:
    print(
        "Usage: python main.py <path to config file>"
        "<OPTIONAL: path to checkpoint>"
    )
    exit()

config_path = sys.argv[1]
with open(config_path, 'r') as ymlfile:
    config = yaml.load(ymlfile)

checkpoint_path = None
if len(sys.argv) > 2:
    checkpoint_path = sys.argv[2]

datagen = CocoStuffDatagen(config)
model = SegNet(1, 3)
trainer = Trainer(datagen, model, config)
trainer.run(checkpoint_path)
