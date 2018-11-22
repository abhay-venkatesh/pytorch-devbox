from datagens.cifar import CifarDatagen
from models.resnet import resnet101
from trainers.trainer import Trainer

datagen = CifarDatagen()
model = resnet101()
trainer = Trainer(datagen, model)
trainer.run()
