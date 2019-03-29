
import torch 
from torch import nn

class Class(nn.Module):
    def __init__(self, classes):
        super(Class, self).__init__()
        self.classes = classes
        self.conv1 = nn.Sequential( #input shape (1,28,28)
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=3) #2x2)
               
         )
        self.fc = nn.Linear(128,self.classes)
    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv1(x)
        x = x.reshape((batch_size,-1))
        x = self.fc(x)
        return x