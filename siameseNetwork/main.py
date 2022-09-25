from random import shuffle
import torch
import torch.nn as nn 
import torchvision.datasets as dsets
from torch.autograd import Variable
from torchvision import transforms
import numpy as np

from utils import create_iterator, contrastive_loss_function, plot_loss
# import network
from network import SiameseNetwork
# from network import 


batchsize=64
train = dsets.MNIST(root='../data/', train=True, download=True)
# test = dsets.MNIST(root='../data/', train=False, transforms = transforms.Composes(
#	[transforms.ToTensor(),]))

indices = np.random.choice(len(train.train_labels.numpy()), 2000, replace=False)
# indices_test = np.random.choice(len(test.test_labels.numpy()), 100, replace=False)


train_iter = create_iterator(train.train_data.numpy()[indices], 
			train.train_labels.numpy()[indices], batchsize)
# test_iter = create_iterator(test.test_data.numpy()[indices_test], 
			#test.test_labels.numpy()[indices_test], batchsize)

#print(torch.utils.data.DataLoader(train_iter, batch_size=1, shuffle=True))
model = SiameseNetwork()
learning_rate = 0.01
momentum = 0.9
criterion = contrastive_loss_function
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

train_loader = torch.utils.data.DataLoader(train_iter, batch_size=batchsize)

# test_loader = torch.utils.data.DataLoader(test_iter, batchsize=batchsize,
											#shuffle=True)

x0, x1, labels = next(iter(train_loader)) 


train_loss = []
epochs = 10
for epoch in range(epochs):
	print(f'Train Epoch: {epoch} -----------------')
	for batch_idx, (x0, x1, labels) in enumerate(train_loader):
		labels = labels.float()
		x0, x1, labels = Variable(x0), Variable(x1), Variable(labels)
		output1, output2 = model(x0, x1)
		loss = criterion(output1, output2, labels)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		train_loss.append(loss.item())
		if batch_idx % batchsize == 0:
			print(f'Epoch: {epoch} \tLoss: {loss.item():.6f}')
			

torch.save(model, 'siamesenet.pth')

plot_loss(train_loss)