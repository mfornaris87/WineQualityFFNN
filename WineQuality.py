import argparse

from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, random_split
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


# Writer will output to ./runs/ directory by default
writer = SummaryWriter()


class FeatureDataset(Dataset):
    def __init__(self, file_name):
        # Read csv file and load row data into variables
        file_out = pd.read_csv(file_name, sep=";")
        x = file_out.iloc[0:4898, 0:11].values
        y = file_out.iloc[0:4898, 11].values - 3

        # Feature Scaling
        sc = StandardScaler()
        x_train = sc.fit_transform(x)
        y_train = y

        # Converting to torch tensor
        self.X_train = torch.tensor(x_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]


feature_set = FeatureDataset('dataset/winequality-white.csv')
num_rows = len(feature_set)
val_percent = 0.01
val_size = int(num_rows * val_percent)
train_size = num_rows - val_size
train_df, val_df = random_split(feature_set, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(
    train_df, batch_size=10, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    val_df, batch_size=1, shuffle=True)


# Defining ANN architecture
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(11, 100)
        self.l2 = nn.Linear(100, 50)
        self.l3 = nn.Linear(50, 7)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x))
        return self.l3(x)


def train(args, model, device, train_loader, optimizer, epoch):

    # Before training the model, it is imperative to call model.train()
    model.train()

    loss_func = nn.CrossEntropyLoss()

    global_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        data = data.view(args.batch_size,  11)
        target = target.view(args.batch_size)
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        global_loss = (global_loss*(batch_idx) + loss.item())/(batch_idx+1)
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    return global_loss


def test(args, model, device, test_loader, epoch):

    # You must call model.eval() before testing the model
    model.eval()

    loss_func = nn.CrossEntropyLoss()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(args.test_batch_size,  11)
            target = target.view(args.test_batch_size)
            output = model(data)
            # sum up batch loss
            test_loss += loss_func(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    acc = 100. * correct / len(test_loader.dataset)

    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), acc))

    return {'test_loss': test_loss, 'acc': acc}


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='FNN')
    parser.add_argument(
        '--batch-size', type=int, default=10, metavar='N',
        help='input batch size for training (default: 64)')
    parser.add_argument(
        '--test-batch-size', type=int, default=1, metavar='N',
        help='input batch size for testing (default: 1000)')
    parser.add_argument(
        '--epochs', type=int, default=100, metavar='N',
        help='number of epochs to train (default: 10)')
    parser.add_argument(
        '--lr', type=float, default=0.05, metavar='LR',
        help='learning rate (default: 0.01)')
    parser.add_argument(
        '--momentum', type=float, default=0.9, metavar='M',
        help='SGD momentum (default: 0.9)')
    parser.add_argument(
        '--no-cuda', action='store_true', default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--seed', type=int, default=1, metavar='S',
        help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval', type=int, default=100, metavar='N',
        help='how many batches to wait before logging training status')
    parser.add_argument(
        '--save-model', action='store_true', default=True,
        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    model = Net().to(device)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        global_loss_train = train(
            args, model, device, train_loader, optimizer, epoch)
        dic = test(args, model, device, test_loader, epoch)

        writer.add_scalars('LOSS', {'loss_train': global_loss_train}, epoch)
        writer.add_scalars('LOSS', {'loss_test': dic['test_loss']}, epoch)
        writer.add_scalar('Accuracy', dic['acc'], epoch)

    if (args.save_model):
        torch.save(model.state_dict(), "fnn.pt")


if __name__ == '__main__':
    main()
