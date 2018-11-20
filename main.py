from datagens.cifar import CifarDatagen
from trainers.trainer import Trainer
from models.resnet import ResNet
from models.resnet import ResidualBlock

datagen = CifarDatagen()
model = ResNet(ResidualBlock, [2, 2, 2])
trainer = Trainer(datagen, model)
trainer.run()
