import torch
import torch.nn as nn


class Trainer:
    def __init__(self, datagen, model):
        self.datagen = datagen
        self.model = model

    def run(self):
        train_loader = self.datagen.train_loader
        test_loader = self.datagen.test_loader
        model = self.model

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # For updating learning rate
        def update_lr(optimizer, lr):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # Train the model
        total_step = len(train_loader)
        curr_lr = learning_rate
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
                    print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(
                        epoch + 1, num_epochs, i + 1, total_step, loss.item()))

            # Decay learning rate
            if (epoch + 1) % 20 == 0:
                curr_lr /= 3
                update_lr(optimizer, curr_lr)

        # Test the model
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Accuracy of the model on the test images: {} %'.format(
                100 * correct / total))

        # Save the model checkpoint
        torch.save(model.state_dict(), 'resnet.ckpt')