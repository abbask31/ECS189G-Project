import pickle

import numpy as np
from matplotlib import pyplot as plt

# loading ORL dataset
if 1:
	f = open('data/stage_3_data/ORL', 'rb')
	data = pickle.load(f)
	f.close()

	train_X_orl = []
	train_y_orl = []

	for instance in data['train']:
		image_matrix = instance['image']
		img = np.array(image_matrix)
		print(img.shape)
		image_label = instance['label']

		train_X_orl.append(image_matrix)
		train_y_orl.append(image_label)



		plt.imshow(image_matrix)
		plt.show()
		print(image_matrix)
		print(image_label)
		# remove the following "break" code if you would like to see more image in the training set
		break
	orl_train_data = {'X': image_matrix, 'y': image_label}
		
	for instance in data['test']:
		image_matrix = instance['image']
		image_label = instance['label']
		plt.imshow(image_matrix)
		plt.show()
		print(image_matrix)
		print(image_label)
		# remove the following "break" code if you would like to see more image in the testing set
		break
		
# loading CIFAR-10 dataset
if 0:
	f = open('data/stage_3_data/CIFAR', 'rb')
	data = pickle.load(f)
	f.close()
	for instance in data['train']:
		image_matrix = instance['image']
		image_label = instance['label']
		plt.imshow(image_matrix)
		plt.show()
		print(image_matrix)
		print(image_label)
		# remove the following "break" code if you would like to see more image in the training set
		break
		
	for instance in data['test']:
		image_matrix = instance['image']
		image_label = instance['label']
		plt.imshow(image_matrix)
		plt.show()
		print(image_matrix)
		print(image_label)
		# remove the following "break" code if you would like to see more image in the testing set
		break

# loading MNIST dataset
if 0:
	f = open('data/stage_3_data/MNIST', 'rb')
	data = pickle.load(f)
	f.close()
	for instance in data['train']:
		image_matrix = instance['image']
		img = np.array(image_matrix)
		print(img.shape)
		image_label = instance['label']
		plt.imshow(image_matrix, cmap='gray')
		plt.show()
		print(image_matrix)
		print(image_label)
		# remove the following "break" code if you would like to see more image in the training set
		break
		
	for instance in data['test']:
		image_matrix = instance['image']
		image_label = instance['label']
		plt.imshow(image_matrix, cmap='gray')
		plt.show()
		print(image_matrix)
		print(image_label)
		# remove the following "break" code if you would like to see more image in the testing set
		break
	
	
	