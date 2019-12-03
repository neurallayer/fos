'''Demo that shows how to use Tensorboard'''
import torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

from fos import Workout
from fos.callbacks import TensorBoardMeter


def train_model():
    '''Demonstrate use of Tensorboard'''

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    model = torchvision.models.resnet50(False)
    # Have ResNet model take in grayscale rather than RGB
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model = model.to("cuda")
    images, _ = next(iter(trainloader))

    # Standard PyTorch features
    writer = SummaryWriter()
    grid = torchvision.utils.make_grid(images)
    writer.add_image('images', grid, 0)

    # Fos features
    workout = Workout(model, F.nll_loss)
    writer.add_graph(model, workout.mover(images))
    workout.fit(trainloader, callbacks=TensorBoardMeter(writer), epochs=5)

    writer.close()


if __name__ == "__main__":
    train_model()
