from PIL import Image
import numpy as np
import os.path
from pycocotools.coco import COCO
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from trainers.utils.logger import Logger


class Batcher:
    def __init__(self, loader, batch_size):
        self.loader = iter(loader)
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        images, labels = [], []
        while len(images) < self.batch_size:
            try:
                image, label = next(self.loader)
                comp_tensor = torch.ones((1), dtype=torch.long)
                if torch.eq(label[1], comp_tensor):
                    images.extend(image)
                    labels.append(label[0])
            except StopIteration:
                break
        if len(images) < self.batch_size:
            raise StopIteration
        else:
            return torch.stack(images), torch.stack(labels)


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

    def write_checkpoint(self, epoch):
        checkpoint_filename = str(epoch) + ".ckpt"
        checkpoint_path = (
            self.experiment_root + "checkpoints/" + checkpoint_filename)
        torch.save(self.model.state_dict(), checkpoint_filename)

    def run(self, checkpoint_path):
        num_epochs = int(self.parameters["epochs"])
        batch_size = int(self.parameters["batch_size"])
        learning_rate = int(self.parameters["learning_rate"])

        if checkpoint_path:
            epochs_done = int(checkpoint_path.split('.')[1].split("/")[3])
            num_epochs -= epochs_done
            self.model.load_state_dict(torch.load(checkpoint_path))

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        total_step = self.datagen.total_steps
        curr_lr = learning_rate
        for epoch in range(num_epochs):
            batcher = Batcher(self.train_loader, batch_size)
            step = 0
            for images, labels in batcher:
                step += 1

                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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

        self.model.eval()

    def test(self, model):
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