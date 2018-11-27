import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pycocotools.coco import COCO


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
            """
            labels = [
                {
                    category_id: Tensor,
                    segmentation: {
                        counts: [
                                "RLE STRING",
                                ...
                                (len(counts) == len(category_id))
                            ],
                        size: [height, width]
                    }
                },
                ...
            ]
            """
            tensor_batch = []
            for label in labels[:1]:
                mask = COCO.annToMask(label)
                tensor = torch.from_numpy(mask).float()
                resized = transforms.Resize([426, 640])(tensor)
                tensor_batch.append(resized)
            return torch.stack(tensor_batch)
            
        def rle_func():
            rows, cols = height, width
            rle_nums = [int(numstring) for numstring in rle_str.split(' ')]
            rle_pairs = np.array(rle_nums).reshape(-1, 2)
            img = np.zeros(rows * cols, dtype=np.uint8)
            for i, length in rle_pairs:
                i -= 1
                img[i:i + length] = 255
            img = img.reshape(cols, rows)
            img = img.T
            return img

        total_step = len(self.train_loader)
        curr_lr = learning_rate
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                # labels = label_transform(labels)
                # print(len(labels))
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
