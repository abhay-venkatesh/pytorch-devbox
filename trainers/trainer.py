from PIL import Image
import numpy as np
import os.path
from pycocotools.coco import COCO
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from utils.logger import Logger


def UnbiasedPULoss(X, A, rho=0.7):
    """ X: outputs
        A: labels
        rho: noise rate """
    X_ = (X - 1).pow(2)
    numer = X_ - (rho * (X.pow(2)))
    frac = (numer / (1 - rho))
    positive_case = frac * A
    zeroth_case = (1 - A) * (X.pow(2))
    loss = positive_case + zeroth_case
    return loss.sum()


class Trainer:
    def __init__(self, datagen, model, config):
        self.datagen = datagen
        self.train_loader = datagen.train_loader
        self.test_loader = datagen.test_loader
        self.device = device = torch.device('cuda' if torch.cuda.
                                            is_available() else 'cpu')
        self.model = model.to(self.device)
        self.config = config
        self.experiment_root = "./experiments/" + experiment_name + "/"
        self.logger = Logger(self.experiment_root)

    def run(self, checkpoint_path):
        def update_lr(optimizer, lr):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        def print_image():
            mask = squeezed.numpy()
            mask_ = np.multiply(mask, 200)
            Image.fromarray(mask_).show()
            seg.show()

        def build_batch(data_itr, images, labels):
            images_ = []
            labels_ = []
            comp_tensor = torch.ones((1), dtype=torch.long)
            if torch.eq(labels[1], comp_tensor):
                images_.extend(images)
                labels_.append(labels[0])
            while len(images_) < batch_size:
                try:
                    images, labels = next(data_itr)
                    comp_tensor = torch.ones((1), dtype=torch.long)
                    if torch.eq(labels[1], comp_tensor):
                        images_.extend(images)
                        labels_.append(labels[0])
                except StopIteration:
                    break
            return images_, labels_

        def write_checkpoint(epoch):
            checkpoint_filename = str(epoch) + ".ckpt"
            checkpoint_path = self.experiment_root + checkpoint_filename
            torch.save(self.model.state_dict(), checkpoint_filename)

        experiment_name = self.config["name"]
        parameters = self.config["parameters"]
        num_epochs = int(parameters["epochs"])
        batch_size = int(parameters["batch_size"])
        learning_rate = int(parameters["learning_rate"])

        if checkpoint_path:
            epochs_done = int(checkpoint_path.split('.')[1].split("/")[3])
            num_epochs -= epochs_done
            self.model.load_state_dict(torch.load(checkpoint_path))

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        total_step = self.datagen.total_steps
        curr_lr = learning_rate
        for epoch in range(num_epochs):
            data_itr = iter(self.train_loader)
            step = 0
            for images, labels in data_itr:
                step += 1
                images_, labels_ = build_batch(data_itr, images, labels)
                if len(images_) < batch_size:
                    break
                images = torch.stack(images_)
                labels = torch.stack(labels_)
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)
                # loss = UnbiasedPULoss(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (step + 1) % 100 == 0:
                    print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(
                        epoch + 1, num_epochs, step + 1, total_step,
                        loss.item()))

                    info = {'loss': loss.item()}
                    self.logger.log(info, step)

            if (epoch + 1) % 20 == 0:
                curr_lr /= 3
                update_lr(optimizer, curr_lr)

            write_checkpoint(epoch)

        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Accuracy of the model on the test images: {} %'.format(
                100 * correct / total))
