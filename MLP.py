import torch
import torch.nn.functional as F   # 激励函数的库
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np

# 定义全局变量
n_epochs = 10     # epoch 的数目
batch_size = 20  # 决定每次读取多少图片


train_data = datasets.MNIST(root = './data', train = True, download = True, transform = transforms.ToTensor())
test_data = datasets.MNIST(root = './data', train = True, download = True, transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, num_workers = 0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, num_workers = 0)


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP,self).__init__()    # 

        self.fc1 = torch.nn.Linear(784,512)
        self.fc2 = torch.nn.Linear(512,128)
        self.fc3 = torch.nn.Linear(128,10)
        
    def forward(self,din):

        din = din.view(-1,28*28)
        dout = F.relu(self.fc1(din))
        dout = F.relu(self.fc2(dout))
        dout = F.softmax(self.fc3(dout), dim=1)

        return dout


def train():

    lossfunc = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params = model.parameters(), lr = 0.01)
    for epoch in range(n_epochs):
        train_loss = 0.0
        for data,target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = lossfunc(output,target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)
        train_loss = train_loss / len(train_loader.dataset)
        print('Epoch:  {}  \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))
        test()


def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 训练集中不需要反向传播
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))
    return 100.0 * correct / total


model = MLP()

if __name__ == '__main__':
    train()
    test()