import argparse
from cProfile import label
import random
import time
import pickle

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.datasets as dsets
import torch.nn as nn
from torch.autograd import Variable
from trochvision import transforms


# The dataset class
class Dataset(object):
	'''
	Class Dataset:
		Input: numpy values
		Output: torch variables
	'''
	def __init__(self, x0, x1, label):
		self.size = label.shape[0]
		self.x0 = torch.from_numpy(x0)
		self.x1 = torch_from_numpy(x1)
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
	ret = Dataset(x0, x1, label)
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

