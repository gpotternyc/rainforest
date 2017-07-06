import glob
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.nn.functional as functional
import torch.optim as optim
from libtiff import TIFF
import argparse
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--disable-cuda", action="store_true", default=False)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--momentum", type=float, default=0.5)
parser.add_argument("--save_every", type=int, default=5)

args = parser.parse_args()
if args.cuda and not torch.cuda.is_available():
	print("no cuda... :(")
	args.cuda = False

class Clouds(data.Data):
	def __init__(self, path="../../dataset/train"):
		self.images = sorted(glob.glob(path+"/*"))
		self.clouds = set(open(path+"../clouds.txt").read().splitlines())
	def __getitem__(self, index):
		img = TIFF.open(self.images[index], mode="r").read_image(),
		return (torch.from_numpy(img).permute(2, 0, 1),
				1 if self.images[index] in self.clouds else 0)
	def __len__(self):
		return len(self.images)

model = models.squeezenet1_0(pretrained=true)
if args.cuda:
	model.cuda()
opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

normalize = transforms.Normalize(
	mean=[1,2,3], std=[4,5,6]
)
dataset = Clouds()
loader = data.DataLoader(dataset)

model.train()
for epoch in range(args.epochs):
	for batch_num, (img, target) in enumerate(loader):
		if args.cuda:
			img = img.cuda()
			target = target.cuda()
		img = normalize(img)
		img = Variable(img); target = Variable(target)
		opt.zero_grad()
		out = model(img)
		loss = functional.binary_cross_entropy(out, target)
		loss.backward()
		opt.step()
		if batch_num % 50 == 0:
			print("Epoch: {}, batch: {}".format(epoch, batch_num))
	if (epoch+1) % args.save_every == 0:
		torch.save(model, epoch)
