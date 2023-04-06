import torchvision
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 使用GPU进行训练，将net和data放在GPU上
random_seed = torch.rand(1)
torch.manual_seed(int(random_seed))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 512, kernel_size=3)
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 5)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.reshape(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x)


def train(epochs):
    for epoch in range(epochs):
        if os.path.exists("./model.pth"):
            network.load_state_dict(torch.load("./model.pth"))
        network.train()
        for batch_idx, (data, target) in enumerate(train_data_loader):
            target = target // 2
            # print(target)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('for epoch {} [{:.0f}%] Cross Entropy Loss is {:.6f}'.format(epoch + 1, 100. * batch_idx / (
                        len(train_data_loader.dataset) // train_batch_size) + 1, loss.item()))
                torch.save(network.state_dict(), './model.pth')
                torch.save(optimizer.state_dict(), './optimizer.pth')


# 设置代理，加速下载
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ["socks_proxy"] = "http://127.0.0.1:7890"

# 下载数据集并构建data_loader，归一化预处理
train_batch_size = 64
test_batch_size = 100
# 加入先验条件，使用均值mean和标准差std对数据集进行归一化
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.1307,), (0.3081,))])
train_data_loader = DataLoader(
    torchvision.datasets.MNIST('./mnist_data', train=True, download=True, transform=transform),
    batch_size=train_batch_size, shuffle=True)
test_data_loader = DataLoader(torchvision.datasets.MNIST('./mnist_data', download=True, transform=transform),
                              batch_size=test_batch_size, shuffle=True)
# print(train_data_loader.dataset)

# examples = enumerate(test_data_loader)
# batch_idx, (example_data, example_targets) = next(examples)
# print(example_targets)
# print(example_data.shape)

learning_rate = 0.03
momentum = 0.5  # 设置动量，逃出局部最小值
n_epochs = 3
log_interval = 10
network = Net()
network.to(device)
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
test_losses = []
test_counter = [i * len(train_data_loader.dataset) for i in range(n_epochs + 1)]

train(3)
