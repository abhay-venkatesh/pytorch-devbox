from datagens.coco import CocoDatagen
from models.segnet import SegNet
from trainers.coco_trainer import Trainer

datagen = CocoDatagen()
model = SegNet(80, 3)
trainer = Trainer(datagen, model)
trainer.run()
