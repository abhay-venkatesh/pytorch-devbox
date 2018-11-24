from datagens.coco import CocoDatagen
from models.deeplab import ASPP
from trainers.trainer import Trainer

datagen = CocoDatagen()
model = ASPP()
trainer = Trainer(datagen, model)
trainer.run()
