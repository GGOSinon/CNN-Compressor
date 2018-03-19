import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class CompressNet(nn.Module):
    def __init__(self):
        super(CompressNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 5)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 10, 5)
        self.out = nn.Linear(10*16*16, 10)

    def forward(self, x):
        #print(x.size())
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = x.view(-1, 10*16*16)
        x = self.out(x)
        #print(x.size())
        return x

    def train(self, inputs, target):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.001) 
        #target = Variable(torch.from_numpy(target))
        #target = target.type(torch.FloatTensor)
        optimizer.zero_grad()
        outputs = self(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        return loss
