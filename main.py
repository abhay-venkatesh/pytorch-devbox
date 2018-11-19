import torchvision.models as models
from datagens.cifar import CifarDatagen
from trainers.trainer import Trainer

# Define datagen
datagen = CifarDatagen()

# Define model
model = models.resnet18()

# Define trainer
trainer = Trainer(datagen, model)

# Execute
trainer.run()
