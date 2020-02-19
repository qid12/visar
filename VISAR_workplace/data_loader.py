import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import rdkit

class compound_dataset(Dataset):
	def __init__(self, list_IDs, data, FP_type):
		self.outputs = outputs
		self.list_IDs = list_IDs

	def featurizer(self, FP_type):
		# generate FPs
		return Xs

	def valid_weight(self):
		return weight

	def __getitem__(self, i):
		return zip(X, y, w)

	def __len__(self):
		return len(self.list_IDs)


def compound_data_loader(fname, smiles_field, output_field, FP_type, 
	                     add_property = None, random_seed = 0):
	df = pd.read_csv(fname, sep = ',')

	# extract clean datasets based on output_field
	df_new = None

	# add additional chemical properties to outputs
	df_new = df_new

	# data preprocessing (including scale and clip, saving the related values to a json file)
	df_new = df_new

	# train test partition
	partition = {'train': [], 'test': []}

	# prepare generator
	train_loader = compound_dataset(partition['train'], df_new, FP_type)
	test_loader = compound_dataset(partition['test'], df_new, FP_type)

	return train_loader, test_loader 


