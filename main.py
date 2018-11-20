from datagens.cifar import CifarDatagen
from trainers.trainer import Trainer
from models.resnet import resnet18
from models.resnet import BasicBlock

datagen = CifarDatagen()
model = resnet18()
trainer = Trainer(datagen, model)
trainer.run()
