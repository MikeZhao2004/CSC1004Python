from __future__ import print_function
import multiprocessing
import random
from statistics import mean

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.backends import mps

from utils.config_utils import read_args, load_config, Dict2Object


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch, processID):
    model.train()
    correct = 0
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        prediction = output.argmax(dim=1, keepdim=True)
        correct += prediction.eq(target.view_as(prediction)).sum().item()
    training_acc = 100.0 * correct / len(train_loader.dataset)
    training_loss = total_loss / len(train_loader)

    print('Process {}    Training epoch #{}, Loss: {:.6f}, Accuracy: {:.2f}%'.format(processID, epoch, training_loss,
                                                                                     training_acc))

    return training_acc, training_loss


def test(model, device, test_loader, epoch, processID):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            prediction = output.argmax(dim=1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100.0 * correct / len(test_loader.dataset)

    print('Process {}    Testing epoch #{}, test Loss: {:.6f}, Accuracy: {:.2f}%'.format(processID, epoch, test_loss,
                                                                                         test_acc))

    return test_acc, test_loss


def plot(epochs, performance, name):
    import matplotlib.pyplot as plt

    plt.title(name)
    plt.plot(epochs, performance, label=name)
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(name + '.png')
    plt.show()


def run(config, processID):
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    use_mps = not config.no_mps and torch.backends.mps.is_available()

    seed = 0
    if processID == 1:
        seed = config.seed1
    elif processID == 2:
        seed = config.seed2
    else:
        seed = config.seed3
    print('Process {} with seed {}'.format(processID, seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': config.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': config.test_batch_size, 'shuffle': True}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True, }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # download data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)

    """add random seed to the DataLoader, pls modify this function"""
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=config.lr)

    """record the performance"""
    epochs = []
    training_accuracies = []
    training_loss = []
    testing_accuracies = []
    testing_loss = []
    training_file = open('training{}.txt'.format(processID), mode='a')
    testing_file = open('testing{}.txt'.format(processID), mode='a')

    scheduler = StepLR(optimizer, step_size=1, gamma=config.gamma)
    for epoch in range(1, config.epochs + 1):
        epochs.append(epoch)
        train_acc, train_loss = train(config, model, device, train_loader, optimizer, epoch, processID)
        training_accuracies.append(train_acc)
        training_loss.append(train_loss)
        test_acc, test_loss = test(model, device, test_loader, epoch, processID)
        testing_accuracies.append(test_acc)
        testing_loss.append(test_loss)
        scheduler.step()
        training_file.write(str(training_loss[epoch-1]) + ' ' + str(training_accuracies[epoch-1]) + '\n')
        testing_file.write(str(testing_loss[epoch-1]) + ' ' + str(testing_accuracies[epoch-1]) + '\n')
        """update the records, Fill your code"""

    """plotting training performance with the records"""
    plot(epochs, training_loss, 'Process {} Training Loss'.format(processID))
    plot(epochs, training_accuracies, 'Process {} Training Accuracy'.format(processID))

    """plotting testing performance with the records"""
    plot(epochs, testing_accuracies, 'Process {} Testing Accuracy'.format(processID))
    plot(epochs, testing_loss, 'Process {} Testing Loss'.format(processID))

    if config.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


def plot_mean():
    training_data = [[], [], []]
    testing_data = [[], [], []]
    mean_training_loss = []
    mean_training_accuracy = []
    mean_testing_loss = []
    mean_testing_accuracy = []
    epochs = []
    for i in range(1, 4):
        with open('training{}.txt'.format(i), mode='r', encoding='utf-8') as training_file:
            for line in training_file:
                training_data[i - 1].append(list(map(float, line.strip('\n').split())))
        with open('testing{}.txt'.format(i), mode='r', encoding='utf-8') as training_file:
            for line in training_file:
                testing_data[i - 1].append(list(map(float, line.strip('\n').split())))

    mean_training_file = open('trainingmean.txt', mode='a')
    mean_testing_file = open('testingmean.txt', mode='a')
    for i in range(len(training_data[0])):
        epochs.append(i+1)
        mean_training_loss.append(mean([training_data[0][i][0], training_data[1][i][0], training_data[2][i][0]]))
        mean_testing_loss.append(mean([testing_data[0][i][0], testing_data[1][i][0], testing_data[2][i][0]]))
        mean_training_accuracy.append(mean([training_data[0][i][1], training_data[1][i][1], training_data[2][i][1]]))
        mean_testing_accuracy.append(mean([testing_data[0][i][1], testing_data[1][i][1], testing_data[2][i][1]]))
        mean_training_file.write(str(mean_training_loss[-1]) + ' ' + str(mean_training_accuracy[-1]) + '\n')
        mean_testing_file.write(str(mean_testing_loss[-1]) + ' ' + str(mean_testing_accuracy[-1]) + '\n')
    mean_training_file.close()
    mean_testing_file.close()

    plot(epochs, mean_training_accuracy, 'Mean Training Accuracy')
    plot(epochs, mean_training_loss, 'Mean Training Loss')
    plot(epochs, mean_testing_accuracy, 'Mean Testing Accuracy')
    plot(epochs, mean_testing_loss, 'Mean Testing Loss')


def clean_file():
    for i in range(1, 4):
        with open('training{}.txt'.format(i), 'w') as f:
            f.truncate(0)
            f.close()
        with open('testing{}.txt'.format(i), 'w') as f:
            f.truncate(0)
            f.close()
    with open('trainingmean.txt', 'w') as f:
        f.truncate(0)
        f.close()
    with open('testingmean.txt', 'w') as f:
        f.truncate(0)
        f.close()


if __name__ == '__main__':
    import torch

    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    # print(f"PyTorch version: {torch.__version__}")
    # print(f"CUDA version: {torch.version.cuda}")
    # print(f"CUDNN version: {torch.backends.cudnn.version()}")
    # print(f"Device name: {torch.cuda.get_device_name(0)}")
    # else:
    #     print("CUDA is not available.")

    arg = read_args()

    """toad training settings"""
    config = load_config(arg)

    clean_file()

    print('MultiProcess activated.')

    processes = []
    for i in range(1, 4):
        process = multiprocessing.Process(target=run, args=(config, i))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()

    plot_mean()
