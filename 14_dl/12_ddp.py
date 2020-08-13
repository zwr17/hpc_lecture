import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import os

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '8888'
rank = int(os.getenv('OMPI_COMM_WORLD_RANK', '0'))
world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', '1'))
dist.init_process_group("nccl", rank=rank, world_size=world_size)
device = torch.device('cuda',rank)

class TwoLayerNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(D_in, H)
        self.fc2 = nn.Linear(H, D_out)

    def forward(self, x):
        x = x.view(-1, D_in)
        h = self.fc1(x)
        h_r = F.relu(h)
        y_p = self.fc2(h_r)
        return F.log_softmax(y_p, dim=1)

epochs = 10
batch_size = 32
D_in = 784
H = 100
D_out = 10
learning_rate = 1.0e-02

# read input data and labels
train_dataset = datasets.MNIST('./data', 
                               train=True, 
                               download=True, 
                               transform=transforms.ToTensor())

validation_dataset = datasets.MNIST('./data', 
                                    train=False, 
                                    transform=transforms.ToTensor())

train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset,
    num_replicas=torch.distributed.get_world_size(),
    rank=torch.distributed.get_rank())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           sampler=train_sampler)

validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, 
                                                batch_size=batch_size, 
                                                shuffle=False)


# define model
model = TwoLayerNet(D_in, H, D_out).to(device)
model = DDP(model, device_ids=[rank])

# define loss function
criterion = nn.CrossEntropyLoss()

# define optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def validate(loss_vector, accuracy_vector):
    model.eval()
    val_loss, correct = 0, 0
    for data, target in validation_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        val_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct.to(torch.float32) / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)
    
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(validation_loader.dataset), accuracy))



lossv, accv = [], []
for epoch in range(epochs):
    # Set model to training mode
    model.train()

    t = time.perf_counter()
    # Loop over each batch from the training set
    for batch_idx, (x, y) in enumerate(train_loader):
        # Copy data to GPU if needed
        x = x.to(device)
        y = y.to(device)

        # forward pass: compute predicted y
        y_p = model(x)

        # compute loss
        loss = criterion(y_p, y)

        # backward pass
        optimizer.zero_grad()
        loss.backward()

        # update weights
        optimizer.step()

        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{:>5}/{} ({:.0%})]\tLoss: {:.6f}\t Time:{:.4f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                batch_idx / len(train_loader), loss.data.item(),
                time.perf_counter() - t))
            t = time.perf_counter()

    validate(lossv, accv)
dist.destroy_process_group()
