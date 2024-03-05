import torch
import matplotlib.pyplot as plt
import numpy as np
from model import PredNet
import torchvision
from torchvision import transforms, datasets
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader, Dataset, Subset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

A_Ahat_out_channels = (3, 48, 96)
R_out_channels = (3, 48, 96)
input_shape = (16, 5, 3, 32, 32)
num_timesteps = input_shape[1]
CrossEntropyLoss = torch.nn.CrossEntropyLoss()

def unnormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul(s).add(m)  # 这里可以使用原地操作，因为我们是在循环内部修改t
    return tensor

def fgsm_attack(image, epsilon, data_grad):
    """x‘ = x + ε * sign(▽L)"""
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon*sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1) # cilp
    return perturbed_image

def test_rotate(image, num_steps=5):
    rotation_step = 90 / num_steps
    rotated_images = [transforms.functional.rotate(image, rotation_step * i) for i in range(num_steps)]
    return rotated_images

class TestRotatedImagesDataset(Dataset):
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
        rotated_images = test_rotate(image, num_steps=5)
        return torch.stack(rotated_images), label

test_set = torchvision.datasets.CIFAR10(
        root='./cifar-10',
        train=False,
        download=True)
rotated_set = TestRotatedImagesDataset(test_set)

indices = {class_id: [] for class_id in range(10)}
for idx, (_, label) in enumerate(rotated_set):
    indices[label].append(idx)

samples_each_class = 100
sampled_indices = []
for ids in indices.values():
    sampled_indices.extend(np.random.choice(ids, samples_each_class, replace=False)) # uniform sampling rotated image sequences
rotated_set = Subset(rotated_set, sampled_indices)

test_loader = DataLoader(rotated_set, batch_size=16, shuffle=False, drop_last=True)

def test_model(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        initial_states = model.init_states(input_shape)  # init E_list in each epoch
        states = initial_states
        for step, data in enumerate(test_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            output, *_ = model(inputs, states)
            test_loss += CrossEntropyLoss(output, labels).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

def adversarial_test(model, device, test_loader, epsilon):
    model.eval()
    correct = 0
    initial_states = model.init_states(input_shape)  # init E_list in each epoch
    states = initial_states
    for step, data in enumerate(test_loader):
        inputs, labels = data[0].to(device), data[1].to(device)
        inputs.requires_grad = True

        # Generate adversarial inputs with FGSM
        output, *_ = model(inputs, states)
        loss = CrossEntropyLoss(output, labels)
        model.zero_grad()
        loss.backward()

        inputs_grad = inputs.grad.data
        perturbed_inputs = fgsm_attack(inputs, epsilon, inputs_grad)

        adv_out, *_ = model(perturbed_inputs, states)
        adv_pred = adv_out.max(1, keepdim=True)[1]
        correct += adv_pred.eq(labels.view_as(adv_pred)).sum().item()
    print('\nAdversarial test set: epsilon: {} Accuracy: {}/{} ({:.0f}%)\n'.format(
        epsilon,  correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

'''
test_images, label = rotated_set[0]
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
imshow(torchvision.utils.make_grid(test_images))
'''


file_name = "./param_dict/" + 'prednet.pth'
prednet = PredNet(A_Ahat_out_channels, R_out_channels, device)
prednet.load_state_dict(torch.load(file_name))
test_model(prednet, device, test_loader)
for epsilon in [0.0, 0.05, 0.1, 0.15, 0.2]:
    adversarial_test(prednet, device, test_loader, epsilon)