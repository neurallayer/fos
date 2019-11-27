import torch
import torch.nn.functional as F
import torchvision
from fos import Workout
from fos.meters import TensorBoardMeter
from torchvision import datasets, transforms

from torch.utils.tensorboard import SummaryWriter

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
model = torchvision.models.resnet50(False)
# Have ResNet model take in grayscale rather than RGB
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model = model.to("cuda")
images, labels = next(iter(trainloader))

grid = torchvision.utils.make_grid(images)


writer = SummaryWriter()

workout = Workout(model, F.nll_loss)

writer.add_image('images', grid, 0)

writer.add_graph(model, workout.mover(images))

workout.fit(trainloader, cb=TensorBoardMeter(writer, metrics=["loss"]))

writer.close()
