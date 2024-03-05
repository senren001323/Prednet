import numpy as np
import matplotlib.pyplot as plt
from model import PredNet
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.dataset import random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
def rotate(image, num_steps=5):
    rotation_step = 90 / num_steps
    rotated_images = [transforms.functional.rotate(image, rotation_step * i) for i in range(num_steps)]
    return rotated_images


class RotatedImagesDataset(Dataset):
    """Generate rotated images for sequence learning"""
    def __init__(self, cifar_10):
        self.cifar_10 = cifar_10
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

    def __len__(self):
        return len(self.cifar_10)

    def __getitem__(self, idx):
        image, label = self.cifar_10[idx]
        image = self.transform(image)
        rotated_images = rotate(image, num_steps=5)
        return torch.stack(rotated_images), label

train_set = torchvision.datasets.CIFAR10(
    root='./cifar-10',
    train=True,
    download=True)
rotated_set = RotatedImagesDataset(train_set)

test_set = torchvision.datasets.CIFAR10(
    root='./cifar-10',
    train=False,
    download=True,
    transform=transform)

# sampled data form rotated dataset
indices = {class_id: [] for class_id in range(10)}
for idx, (_, label) in enumerate(rotated_set):
    indices[label].append(idx)

samples_each_class = 2000
sampled_indices = []
for ids in indices.values():
    sampled_indices.extend(np.random.choice(ids, samples_each_class, replace=False)) # uniform sampling rotated image sequences
rotated_set = Subset(rotated_set, sampled_indices)

num_train = int(len(rotated_set) * 0.95)  # 95% for training, 5% for validation
num_validation = len(rotated_set) - num_train
train, validation = random_split(rotated_set, [num_train, num_validation])

train_loader = DataLoader(train, batch_size=16, shuffle=True, drop_last=True)
validation_loader = DataLoader(validation, batch_size=16, shuffle=False, drop_last=True)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False, drop_last=True)

# model hyperparameters
A_Ahat_out_channels = (3, 48, 96)
R_out_channels = (3, 48, 96)
input_shape = (16, 5, 3, 32, 32)
num_timesteps = input_shape[1]
prednet_lr = 1e-3
epochs = 30
CrossEntropyLoss = torch.nn.CrossEntropyLoss()

# create model, optimizer, scheduler
prednet = PredNet(A_Ahat_out_channels, R_out_channels, device).to(device)
prednet_optimizer = torch.optim.Adam(prednet.parameters(), lr=prednet_lr)
lr_maker = torch.optim.lr_scheduler.StepLR(prednet_optimizer, step_size=30, gamma=0.1)

# debug info
init_states = prednet.init_states(input_shape)
print(prednet)
print(f'initial length: {len(init_states)}')
for i in range(len(init_states)):
    print(f'initial {i}: {init_states[i].shape}')

def validate_model(model, device, validation_loader):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        initial_states = model.init_states(input_shape)  # init E_list in each epoch
        states = initial_states
        for step, data in enumerate(validation_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            output, *_ = model(inputs, states)
            val_loss += CrossEntropyLoss(output, labels).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()

    val_loss /= len(validation_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(validation_loader.dataset), 100. * correct / len(validation_loader.dataset)))

torch.autograd.set_detect_anomaly(True)

for ep in range(epochs):
    tr_loss = 0
    initial_states = prednet.init_states(input_shape) # init E_list in each epoch
    states = initial_states
    for step, data in enumerate(train_loader):
        inputs, labels = data[0].to(device), data[1].to(device)
        A_0 = inputs
        output, output_list, frame_list = prednet(A_0, states) # E_list = next states
        #print(step, output.shape)

        prednet_optimizer.zero_grad()
        layer_weights = np.array([0.1 for _ in range(prednet.num_layers)])
        layer_weights[0] = 1.0
        layer_weights = torch.FloatTensor(layer_weights).to(device)
        error_list = [batch_error * layer_weights for batch_error in output_list] # weight all layers' E
                                                                                  # List of [(batch_size, nums_layer)], length is timesteps
        time_weight = (1.0 / (num_timesteps-1))
        time_weight = [time_weight for _ in range(num_timesteps-1)]
        time_weight.insert(0, 0.0)
        time_weight = torch.FloatTensor(time_weight).to(device)
        error_list = [error_t.sum() for error_t in error_list] # summing every timestep's weight. List of [flaot, ...], length is timesteps

        all_error = error_list[0] * time_weight[0] # equally weight all timesteps except the first
        for error, weight in zip(error_list[1:], time_weight[1:]):
            all_error = all_error + error * weight
        cross_loss = CrossEntropyLoss(output, labels)
        total_loss = cross_loss
        total_loss.backward()
        prednet_optimizer.step()

        tr_loss += total_loss
    lr_maker.step()
    print(f'Epoch {ep + 1}, Training Loss: {tr_loss / len(train_loader)}')
    validate_model(prednet, device, validation_loader)

file_name = "./param_dict/" + 'prednet.pth'
torch.save(prednet.state_dict(), file_name)