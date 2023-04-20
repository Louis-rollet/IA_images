import torch
import torchvision
import gradio
import matplotlib


class Model:
    def __init__(self, img_size, num_channels, num_kernel):
        self.conv1 = torch.nn.Conv2d(img_size * img_size * num_channels, num_channels, num_kernel)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(num_channels, 16, num_kernel)
        self.linear1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.linear2 = torch.nn.Linear(120, 84)
        self.linear3 = torch.nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    def train(self, train_loader, epoch, learning_rate):
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        for i in range(epoch):
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % 100 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        i, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
                        
