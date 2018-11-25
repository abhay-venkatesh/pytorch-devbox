import torch
import torch.nn as nn
import torchvision.transforms as transforms


class Trainer:
    def __init__(self, datagen, model):
        self.train_loader = datagen.train_loader
        self.test_loader = datagen.test_loader
        self.device = device = torch.device('cuda' if torch.cuda.
                                            is_available() else 'cpu')
        self.model = model.to(self.device)

    def run(self, num_epochs=80, learning_rate=0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        def update_lr(optimizer, lr):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        def label_transform(labels):
            tensor_list = labels[0]['bbox']
            print(tensor_list)
            return label_tensor

        total_step = len(self.train_loader)
        curr_lr = learning_rate
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = label_transform(labels)
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
