
from datagens.cocobbox import CocoBboxDatagen
from models.segnet import SegNet
from trainers.cocobbox_trainer import Trainer
import sys
import torchvision.models as models
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

datagen = CocoBboxDatagen(config)
vgg16 = models.vgg16(pretrained=True)
model = SegNet(2, 3)
model.init_vgg16_params(vgg16)
trainer = Trainer(datagen, model, config)
trainer.run(checkpoint_path)
