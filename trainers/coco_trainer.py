import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from PIL import Image
import numpy as np


class Trainer:
    def __init__(self, datagen, model):
        self.train_loader = datagen.train_loader
        self.test_loader = datagen.test_loader
        self.device = device = torch.device('cuda' if torch.cuda.
                                            is_available() else 'cpu')
        self.model = model.to(self.device)

    def run(self, batch_size=3, num_epochs=80, learning_rate=0.001):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        def update_lr(optimizer, lr):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        def print_image():
            mask = squeezed.numpy()
            print(mask)
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

        total_step = len(self.train_loader)
        curr_lr = learning_rate
        for epoch in range(num_epochs):
            data_itr = iter(self.train_loader)
            i = 0
            for images, labels in data_itr:
                i += 1
                images_, labels_ = build_batch(data_itr, images, labels)
                if len(images_) < batch_size:
                    break
                images = torch.stack(images_)
                labels = torch.stack(labels_)
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 100 == 0:
                    print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(
                        epoch + 1, num_epochs, i + 1, total_step, loss.item()))

            if (epoch + 1) % 20 == 0:
                curr_lr /= 3
                update_lr(optimizer, curr_lr)

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

        torch.save(self.model.state_dict(), 'resnet.ckpt')
