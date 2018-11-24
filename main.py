from datagens.coco import CocoDatagen
from models.segnet import SegNet
from trainers.trainer import Trainer

datagen = CocoDatagen()
model = SegNet(2, 3)
trainer = Trainer(datagen, model)
trainer.run()
