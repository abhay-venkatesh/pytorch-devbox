from PIL import Image
from pycocotools.coco import COCO
from trainers.utils.logger import Logger
import numpy as np
import os.path
import torch
import torch.nn as nn
import torchvision.transforms as transforms


class Trainer:
    def __init__(self, datagen, model, config):
        self.config = config
        self.datagen = datagen
        self.device = torch.device('cuda' if torch.cuda.
                                   is_available() else 'cpu')
        self.experiment_name = self.config["name"]
        self.experiment_root = "./experiments/" + self.experiment_name + "/"
        self.log_path = self.experiment_root + "/logs/"
        self.logger = Logger(self.log_path)
        self.model = model.to(self.device)
        self.parameters = self.config["parameters"]
        self.test_loader = self.datagen.test_loader
        self.train_loader = self.datagen.train_loader

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

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        total_step = len(train_loader)
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                        epoch + 1, num_epochs, i + 1, total_step, loss.item()))

            self.write_checkpoint(epoch)

    def test(self, model):
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