import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt


# Section A Hyper Parameters
input_size = 784
num_classes = 10
num_epochs_A = 20
batch_size_A = 100
learning_rate_A = 1e-3


# Section B Hyper Parameters
num_epochs_B = 2
batch_size_B = 100
learning_rate_B = 8e-2

# Section C Hyper Parameters
num_epochs_C = 2
batch_size_C = 32
learning_rate_C = 4e-2
hidden_layer_size = 500

num_epochs_for_eval = 120
batch_size_for_test = 100

# MNIST Dataset
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader_A = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size_A,
                                           shuffle=True)
train_loader_B = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size_B,
                                           shuffle=True)
train_loader_C = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size_C,
                                           shuffle=True)


test_loader_A = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size_A,
                                          shuffle=False)

test_loader_B = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size_B,
                                          shuffle=False)

test_loader_C = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size_C,
                                          shuffle=False)

# Neural Network Model
class Net(nn.Module):
    def __init__(self, input_size, num_classes, section, hidden_layer_size=0):
        super(Net, self).__init__()
        self.section = section
        if self.section == 'A' or self.section == 'B':
            self.fc1 = nn.Linear(input_size, num_classes)
        else:
            self.fc1 = nn.Linear(input_size, hidden_layer_size)
            self.fc2 = nn.Linear(hidden_layer_size, num_classes)



    def forward(self, x):
        if self.section == 'A' or self.section == 'B':
            out = self.fc1(x)
        else:
            x = nn.functional.relu(self.fc1(x))
            out = self.fc2(x)
        return out


net_A = Net(input_size, num_classes, 'A')
net_B = Net(input_size, num_classes, 'B')
net_C = Net(input_size, num_classes, 'C', hidden_layer_size)

if torch.cuda.is_available():
    net_A.cuda()
    net_B.cuda()
    net_C.cuda()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer_A = torch.optim.SGD(net_A.parameters(), lr=learning_rate_A)
optimizer_B = torch.optim.Adagrad(net_B.parameters(), lr=learning_rate_B)
optimizer_C = torch.optim.Adagrad(net_C.parameters(), lr=learning_rate_C)

# Train the Model Function
def train_epoch(num_epochs, model, train_loader, optimizer, section):
    losses = []
    for epoch in range(num_epochs):
        ep_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            # Convert torch tensor to Variable
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            else:
                images = Variable(images.view(-1, 28 * 28))
                labels = Variable(labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            ep_loss += criterion(output, labels).data[0]  # sum up batch loss
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0 or (i+1) == len(train_loader_A):
                print('==>>> section {}, epoch: {}, batch index: {}, train loss: {:.6f}'.format(
                    section, epoch + 1, i + 1, loss.data[0]))
        losses.append(ep_loss / len(train_loader.dataset))
    return losses

# Test the Model Function
def test_epoch(model, test_loader, section):
    correct = 0
    total = 0
    for images, labels in test_loader:
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images.view(-1, 28 * 28))
            labels = Variable(labels)
        output = model(images)
        _, pred_label = torch.max(output.data, 1)
        correct += (pred_label == labels.data).sum()
        total += labels.size(0)

    print('Test set: Section {}, Accuracy: {}/{} ({:.0f}%)'.format(section, correct, total, 100. * correct / total))

#Train all sections
losses_sec_A = train_epoch(num_epochs_for_eval, net_A, train_loader_A, optimizer_A, 'A')
losses_sec_B = train_epoch(num_epochs_for_eval, net_B, train_loader_B, optimizer_B, 'B')
losses_sec_C = train_epoch(num_epochs_for_eval, net_C, train_loader_C, optimizer_C, 'C')

#Test all sections
test_epoch(net_A, test_loader_A, 'A')
test_epoch(net_B, test_loader_B, 'B')
test_epoch(net_C, test_loader_C, 'C')

legend = []
epochs = [i for i in range(num_epochs_for_eval)]
plt.figure(2)
plt.plot(epochs, losses_sec_A, 'b')
legend.append('Section A')
plt.plot(epochs, losses_sec_B, 'g')
legend.append('Section B')
plt.plot(epochs, losses_sec_C, 'r')
legend.append('Section C')

plt.legend(legend, loc=1)
# plt.xscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Epoch')
img_save = 'Q3'
plt.savefig(img_save)

# Save the Models
torch.save(net_A.state_dict(), 'model_section_A.pkl')
torch.save(net_B.state_dict(), 'model_section_B.pkl')
torch.save(net_C.state_dict(), 'model_section_C.pkl')
