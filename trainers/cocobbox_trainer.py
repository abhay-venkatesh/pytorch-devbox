from PIL import Image
from pycocotools.coco import COCO
from trainers.trainer import TrainerBase
from trainers.utils.logger import Logger
import numpy as np
import os.path
import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=-1):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


def UnbiasedPULoss(X, A, rho=0.7):
    """ Note: This must be constrained.
    
        X: outputs
        A: labels
        rho: noise rate """
    X_ = (X - 1).pow(2)
    numer = X_ - (rho * (X.pow(2)))
    frac = (numer / (1 - rho))
    positive_case = frac * A
    zeroth_case = (1 - A) * (X.pow(2))
    loss = positive_case + zeroth_case
    return loss.sum()


class Trainer(TrainerBase):
    def __init__(self, datagen, model, config):
        self.datagen = datagen
        self.train_loader = datagen.train_loader
        self.test_loader = datagen.test_loader
        self.device = torch.device('cuda' if torch.cuda.
                                   is_available() else 'cpu')
        self.model = model.to(self.device)
        self.config = config
        self.parameters = self.config["parameters"]
        self.experiment_name = self.config["name"]
        self.experiment_root = "./experiments/" + self.experiment_name + "/"
        self.log_path = self.experiment_root + "/logs/"
        self.logger = Logger(self.log_path)

    def update_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def print_image(self):
        mask = squeezed.numpy()
        mask_ = np.multiply(mask, 200)
        Image.fromarray(mask_).show()
        seg.show()

    def run(self, checkpoint_path):
        num_epochs = int(self.parameters["epochs"])
        learning_rate = int(self.parameters["learning_rate"])

        num_epochs -= self.load_checkpoint(checkpoint_path)

        criterion = CrossEntropyLoss2d()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        total_step = len(self.train_loader)
        curr_lr = learning_rate
        for epoch in range(num_epochs):
            step = 0
            for images, labels in self.train_loader:
                step += 1

                bboxes = labels[0].long().squeeze(1)

                images = images.to(self.device)
                bboxes = bboxes.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, bboxes)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(
                    epoch + 1, num_epochs, step + 1, total_step,
                    loss.item()))

                if (step + 1) % 100 == 0:
                    print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(
                        epoch + 1, num_epochs, step + 1, total_step,
                        loss.item()))

                    info = {'loss': loss.item()}
                    self.logger.log(self.model, info, step)

            if (epoch + 1) % 20 == 0:
                curr_lr /= 3
                self.update_lr(optimizer, curr_lr)

            self.write_checkpoint(epoch)

    def test(self, model):
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.test_loader:
                labels = labels[0]

                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Accuracy of the model on the test images: {} %'.format(
                100 * correct / total))
