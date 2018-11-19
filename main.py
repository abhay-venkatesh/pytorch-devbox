import torchvision.models as models
from datagens import CifarDatagen
from trainers import Trainer

# Define datagen
datagen = CifarDatagen()

# Define model
model = models.resnet18()

# Define trainer
trainer = Trainer(datagen, model)

# Execute
trainer.run()
