import torch
import torchvision
import gradio
import matplotlib
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy
import matplotlib.pyplot as plt
import numpy as np
import threading



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5) 
        self.linear1 = nn.Linear(16 * 5 * 5, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    def train(self, train_loader, epoch, learning_rate, test_loader):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        for epoch in range(epoch):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 2000 == 1999:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total
    def predict(self, image, classes):
        # image = transforms.ToTensor()(image).unsqueeze(0)
        prediction = self(image)
        return {classes[i]: float(prediction[0][i] / len(prediction[0]))
        for i in range(len(classes)) if prediction[0][i] / len(prediction[0]) > 0.1}
    
    def get_mean_and_std(self, dataset):
        channels_sum = 0
        channels_squared_sum = 0
        batch_num = 0
        print("Request !")
        for image in dataset:
            channels_sum += torch.mean(image[0], dim=[1,2])
            channels_squared_sum += torch.mean(image[0]**2, dim=[1,2])
            batch_num += 1
        mean = channels_sum / batch_num
        std = (channels_squared_sum / batch_num - mean**2)**0.5
        return mean, std
    def normalize_image(self, image, mean, std):
        image = image.permute(2, 0, 1)
        image = (image - mean[:, None, None]) / std[:, None, None]
        return image


class App:
    def __init__(self, model, epoch, batch_size, learning_rate):
        self.model = model
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.train_set = None
        self.train_loader = None
        self.test_set = None
        self.test_loader = None
        self.transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 
        self.add_dataset('./app/dataset')
    def add_dataset(self, dataset_path):
        self.train_set = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=self.transform)
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.test_set = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=self.transform)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)
    def train(self):
        a = self.model.train(self.train_loader, self.epoch, self.learning_rate, self.test_loader)
        print("precision : ", a)
        return self.model
    def predict(self, image):
        return self.model.predict(image, self.classes)
    def show_image(self, image):
        plt.imshow(image)
        plt.show()
    def execute_prediction(self, image):
        threading.Thread(target=self.show_image, args=(image,)).start()
        image = reformate_image(image)
        mean, std = model.get_mean_and_std(self.test_loader)
        image = model.normalize_image(image, mean, std)
        result = self.predict(image)
        print(result)
        return result


def reformate_image(image):
    image = transforms.ToPILImage()(np.uint8(image))
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    
    image = transform(image)
    image = image.permute(1, 2, 0)
    return image


model = Model()
print('Loading model...')
model.load_state_dict(torch.load('./app/model/model.pth'))
print('Model loaded')
app = App(model, 1, 4, 0.001)
# print("Model trained")
# print('Saving model...')
# torch.save(model.state_dict(), './app/model/model.pth')
# print('Model saved')
print('Starting gradio...')
gradio.Interface(app.execute_prediction, "image", "label", title="My_AI", description="Classify images into 10 classes").launch(share=True)
