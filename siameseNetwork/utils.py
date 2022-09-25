import random
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset


# The dataset class
class MNISTDataset(Dataset):
	'''
	Class Dataset:
		Input: numpy values
		Output: torch variables
	'''
	def __init__(self, x0, x1, label):
		self.size = label.shape[0]
		self.x0 = torch.from_numpy(x0)
		self.x1 = torch.from_numpy(x1)
		self.label = torch.from_numpy(label)

	def __getitem__(self, index):
		return (self.x0[index],
					self.x1[index],
					self.label[index])

	def __len__(self):
		return self.size

# Creating pairs function and preprocessing images in them
def create_pairs(data, digit_indices):
	x0_data = []
	x1_data = []
	label = []

	n = min([len(digit_indices[d]) for d in range(10)]) - 1
	for d in range(10): # We have 10 digits in MNIST
		for i in range(n):
			z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
			x0_data.append(data[z1] / 255.)
			x1_data.append(data[z2] / 255.)
			label.append(1)

			inc = random.randrange(1, 10)
			dn = (d + inc) % 10
			z1, z2 = digit_indices[d][1], digit_indices[dn][i]
			x0_data.append(data[z1] / 255.)
			x1_data.append(data[z2] / 255.)
			label.append(0)

	x0_data = np.array(x0_data, dtype=np.float32)
	x0_data = x0_data.reshape([-1, 1, 28, 28]) 
	x1_data = np.array(x1_data, dtype=np.float32)
	x1_data = x1_data.reshape([-1, 1, 28, 28]) 
	label = np.array(label, dtype=np.int32)
	return x0_data, x1_data, label


# Creating the iterator function 
def create_iterator(data, label, batchsize, shuffle=False):
	digit_indices = [np.where(label == i)[0] for i in range(10)]
	x0, x1, label = create_pairs(data, digit_indices)
	ret = MNISTDataset(x0, x1, label)
	return ret


# Creating the loss function 
def contrastive_loss_function(x0, x1, y, margin=1.0):
	# Euclidean distance
	diff = x0 - x1
	dist_sq = torch.sum(torch.pow(diff, 2), 1)
	dist = torch.sqrt(dist_sq)
	mdist = margin - dist
	dist = torch.clamp(mdist, min=0.0)
	loss = y * dist_sq + (1-y) * torch.pow(dist, 2)
	loss = torch.sum(loss) / 2.0 / x0.size()[0]
	return loss

def plot_loss(train_loss, name='train_loss.png'):
	plt.plot(train_loss, label='train loss')
	plt.legend()
	plt.show()


def test_model(model, test_loader):
	model.eval()
	all_data, all_labels = [], []

	with torch.no_grad():
		for batch_idx, (x, labels) in enumerate(test_loader):
			x, labels = Variable(x), Variable(labels)

			output = model.forward_once(x)

			all_data.extend(output.data.cpu().numpy().tolist())
			all_labels.extend(labels.data.cpu().numpy().tolist())
	
	np_all = np.array(all_data)
	np_labels = np.array(all_labels)
	return np_all, np_labels

def testing_plots(model, test_loader):
	dict_pickle = {}
	np_all, np_labels = test_model(model, test_loader)
	dict_pickle['np_all'] = np_all
	dict_pickle['np_labels'] = np_labels

	plot_mnist(np_all, np_labels)


def plot_mnist(np_all, np_labels, name='embeddings_plot.png'):
	c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
		'#ff00ff', '#990000', '#999900', '#009900', '#009999']
	
	for i in range(10):
		f = np_all[np.where(np_labels == i)]
		plt.plot(f[:, 0], f[:, 1], '.', c=c[i])
	plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
	plt.savefig(name)
