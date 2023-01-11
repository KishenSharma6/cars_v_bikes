import torch
from torch.nn import Module, functional as F


class ConvNet(Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, 
                                    out_channels= 10, 
                                    kernel_size= 3,
                                    padding= 1,
                                    stride=1)

        self.pool1 =  torch.nn.MaxPool2d(kernel_size=3, 
                                        stride=1)

        self.batchnorm1 = torch.nn.BatchNorm2d(num_features= 10)

        self.conv2 = torch.nn.Conv2d(in_channels=10, 
                                    out_channels= 16,
                                    kernel_size= 3,
                                    padding=1,
                                    stride=1)

        self.pool2 =  torch.nn.MaxPool2d(kernel_size=3,
                                        stride= 3) #output 42 x 42

        self.batchnorm2 = torch.nn.BatchNorm2d(num_features=16)

        self.conv3 = torch.nn.Conv2d(in_channels=16, 
                                    out_channels= 20,
                                    kernel_size= 3,
                                    padding= 1,
                                    stride= 1)


        self.batchnorm3 = torch.nn.BatchNorm2d(num_features=20)
        #Drop out?
        self.fc1 = torch.nn.Linear(20 * 42 * 42, 512)
        self.fc2 = torch.nn.Linear(512, 128)
        self.fc3 = torch.nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.batchnorm1(self.conv1(x))))
        x = self.pool2(F.relu(self.batchnorm2(self.conv2(x))))
        x = F.relu(self.batchnorm3(self.conv3(x)))

        x= x.flatten()
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    
def train(model, dataloader, criterion, epochs, device, optimizer):
    epoch_loss = []

    #move model to device
    model = model.to(device)

    for epoch in range(epochs):
        running_loss = 0.0
        
        for i, data in enumerate(dataloader):
            #move data to device
            inputs, labels = data['image'].to(device), data['label'].to(device)
            inputs, labels = inputs.to(device), labels.to(device)

            #zero gradients in network
            optimizer.zero_grad()

            #convert input images to scores
            scores = model(inputs)

            #calculate loss for batch
            loss = criterion(scores, labels) #do i need to convert raw scores?

            #calculate the gradient of the loss wrt eacj parameter
            loss.backward()

            #step in direction of the gradient
            optimizer.step()

            running_loss += loss.item()
        epoch_loss.append(running_loss / i)

        print("Epoch " + str(epoch + 1) + " loss:" + str(epoch_loss[epoch]))
    
    print("training complete")
    return epoch_loss
    