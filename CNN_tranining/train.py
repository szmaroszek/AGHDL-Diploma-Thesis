#%%
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy 


def load_data(dataset, x, batch_size):
    validate_size = int(len(dataset)*(x/100))
    train_size = len(dataset) - validate_size

    train_ds, validate_ds = torch.utils.data.random_split(dataset, [train_size, validate_size])

    return (
        torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True),
        torch.utils.data.DataLoader(validate_ds, batch_size, num_workers=4, pin_memory=True)
        )


class CnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),

            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(128*32*32, 1024),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.Dropout(),
            nn.Linear(256, 8)
        )
            
    def forward(self, xb):
        return self.network(xb)


def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
            
    def __iter__(self):
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


def plot_losses(train_losses, val_losses, accuracy_train, accuracy_validate):
    fig, axs = plt.subplots(2)
    axs[0].plot(train_losses, '-bx')
    axs[0].plot(val_losses, '-rx')
    axs[0].set(ylabel='loss')
    axs[0].set_title('Loss & Accuracy vs. No. of epochs')
    
    axs[1].plot(accuracy_train, '-bx')
    axs[1].plot(accuracy_validate, '-rx')
    axs[1].set(xlabel='epochs', ylabel='accuracy')


def train(n_epochs, train_loader, validate_loader, model, optimizer, criterion):
    train_loss_plot = []
    valid_loss_plot = []
    accuracy_validate_plot = []
    accuracy_train_plot = []
    correct = 0.
    total = 0.
    accuracy_validate = 0.
    accuracy_train = 0.
    for epoch in range(1, n_epochs+1):
        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        for batch_idx, (inputs, labels) in enumerate(train_loader):

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss = train_loss + (1 / (batch_idx + 1)) * (loss.data - train_loss)
            _, predicted = torch.max(output, 1)
            correct += (predicted == labels).sum().item()
            total += inputs.size(0)
            accuracy_train = 100. * correct / total
        
        model.eval()
        for batch_idx, (inputs, labels) in enumerate(validate_loader):

            output = model(inputs)
            loss = criterion(output, labels)
            valid_loss = valid_loss + (1 / (batch_idx + 1)) * (loss.data - valid_loss)
            _, predicted = torch.max(output, 1)
            correct += (predicted == labels).sum().item()
            total += inputs.size(0)
            accuracy_validate = 100. * correct / total

        scheduler.step(valid_loss)

        accuracy_validate_plot.append(accuracy_validate)
        accuracy_train_plot.append(accuracy_train)
        valid_loss_plot.append(valid_loss)
        train_loss_plot.append(train_loss)
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))
        
    plot_losses(train_loss_plot, valid_loss_plot, accuracy_train_plot, accuracy_validate_plot)

    return model


def test(loaders, model, criterion):

    test_loss = 0.
    correct = 0.
    total = 0.
    accuracy = 0.
    model.eval()
    class_correct = list(0. for i in range(8))
    class_total = list(0. for i in range(8))

    for batch_idx, (inputs, labels) in enumerate(test_loader):

        output = model(inputs)
        loss = criterion(output, labels)
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        _, predicted = torch.max(output, 1)
        correct += (predicted == labels).sum().item()
        total += inputs.size(0)
        accuracy = 100. * correct / total

        try:
            with torch.no_grad():
                c = (predicted == labels).squeeze()
                for i in range(8):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        except:
            pass
        try:
            print('Accuracy: {}'.format(accuracy))
        except:
            pass
            
    print('Test Loss: {:.6f}\n'.format(test_loss))
    print('\nTest Accuracy: {}% ({}/{})'.format(accuracy, correct, total))

    for i in range(8):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':

    dataset = torchvision.datasets.ImageFolder(
        root=r'###',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(10),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.54334027, 0.5077933, 0.47069696), (0.2848591, 0.28253064, 0.28779432))])
        )

    classes = dataset.classes

    train_loader, validate_loader = load_data(dataset=dataset, x=15, batch_size=10)

    test_ds = torchvision.datasets.ImageFolder(
        root=r'###',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.ToTensor(),])
        )
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=10, num_workers=4, pin_memory=True)

    net = CnnModel()
    
    device = torch.device('cuda')
    
    train_loader = DeviceDataLoader(train_loader, device)
    validate_loader = DeviceDataLoader(validate_loader, device)
    test_loader = DeviceDataLoader(test_loader, device)
    to_device(net, device)

    summary(net, (3, 256, 256))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.0003, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    path = r'###'
    net = train(50, train_loader, validate_loader, net, optimizer, criterion)
    torch.save(net.state_dict(), path)

    test(test_loader, net, criterion)
