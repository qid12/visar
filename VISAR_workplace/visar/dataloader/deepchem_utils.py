import numpy as np
import pandas as pd
import deepchem as dc
import os

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import backend as K

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from itertools import cycle
import matplotlib.colors as colors

from rdkit import Chem
from rdkit.Chem import PandasTools

import cairosvg
from scipy.stats import pearsonr

from visar.utils.visar_utils import extract_clean_dataset

import warnings
warnings.filterwarnings("ignore")
#-------------------------------------

def prepare_dataset(para_dict):
    '''
    input: fname --- name of the file of raw data containing chemicals and the value for each assay;
           task --- list of task names (supporting 1 or more, but must be a list)
           dataset_file --- name of the temperary file saving intermediate dataset containing only the data for a specific training task
           FP_type --- name of the fingprint type (one of 'Circular_2018', 'Circular_1024', 'Morgan', 'MACCS', 'RDKit_FP')
           model_flag --- type of the model (ST: single task, len(task)==1; MT: multitask, len(task)>=1)
           add_features --- must be None for ST; list of properties
           smiles_field --- the name of the field in fname defining the SMILES for chemicals
           id_field --- the name of the field in fname defining the identifier for chemicals
    output: dataset build by deepchem
    '''
    fname = para_dict['dataset_file']
    smiles_field = para_dict['smiles_field']
    id_field = para_dict['id_field']
    task = para_dict['task_list']
    dataset_file = 'tmp.csv'
    model_flag = para_dict['model_flag']
    add_features = para_dict['add_features']
    FP_type = para_dict['feature_type']

    MT_df = pd.read_csv(fname)
    if model_flag == 'ST':
        df = extract_clean_dataset(task, MT_df, smiles_field = smiles_field, id_field = id_field)
    elif model_flag == 'MT':
        df = extract_clean_dataset(task, MT_df, add_features = add_features, smiles_field = smiles_field, id_field = id_field)
        if not add_features is None:
            task = task + add_features
    
    if FP_type == 'Circular_2048':
        df.to_csv(dataset_file)
        featurizer = dc.feat.CircularFingerprint(size=2048)
        loader = dc.data.CSVLoader(id_field=id_field, 
                                   smiles_field=smiles_field, 
                                   tasks = task,
                                   featurizer=featurizer)
        dataset = loader.featurize(dataset_file)
        
    elif FP_type == 'Circular_1024':
        df.to_csv(dataset_file)
        featurizer = dc.feat.CircularFingerprint(size=1024)
        loader = dc.data.CSVLoader(id_field=id_field, 
                                   smiles_field=smiles_field, 
                                   tasks = task,
                                   featurizer=featurizer)
        dataset = loader.featurize(dataset_file)
    
    elif FP_type == 'RDKit_FP':
        df.to_csv(dataset_file)
        featurizer = dc.feat.RDKitDescriptors()
        loader = dc.data.CSVLoader(id_field=id_field, 
                                   smiles_field=smiles_field, 
                                   tasks = task,
                                   featurizer=featurizer)
        dataset = loader.featurize(dataset_file)
        
    elif FP_type == 'Morgan': #2048
        smiles = df[smiles_field].tolist()
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        fps = []
        for mol in mols:
            fps.append(np.matrix(AllChem.GetMorganFingerprintAsBitVect(mol,2)))
        fps = np.concatenate(fps)
        new_df = pd.DataFrame(fps)
        new_df.columns = ['V' + str(i) for i in range(fps.shape[1])]
        new_df[id_field] = df[id_field]
        for m in range(len(task)):
            new_df[task[m]] = df[task[m]]
        new_df.to_csv(dataset_file)
        user_specified_features = ['V' + str(i) for i in range(fps.shape[1])]
        featurizer = dc.feat.UserDefinedFeaturizer(user_specified_features)
        loader = dc.data.UserCSVLoader(
                tasks=task, smiles_field=smiles_field, id_field=id_field,
                featurizer=featurizer)
        dataset = loader.featurize(dataset_file)
        
    elif FP_type == 'MACCS': #167
        smiles = df[smiles_field].tolist()
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        fps = []
        for mol in mols:
            fps.append(np.matrix(MACCSkeys.GenMACCSKeys(mol)))
        fps = np.concatenate(fps)
        new_df = pd.DataFrame(fps)
        new_df.columns = ['V' + str(i) for i in range(fps.shape[1])]
        new_df[id_field] = df[id_field]
        for m in range(len(task)):
            new_df[task[m]] = df[task[m]]
        new_df.to_csv(dataset_file)
        user_specified_features = ['V' + str(i) for i in range(fps.shape[1])]
        featurizer = dc.feat.UserDefinedFeaturizer(user_specified_features)
        loader = dc.data.UserCSVLoader(
                tasks = task, smiles_field=smiles_field, id_field=id_field,
                featurizer=featurizer)
        dataset = loader.featurize(dataset_file)
    
    else:
        print('Unsupported Fingerprint type!')

    os.system('rm %s' % dataset_file)

    # train test split
    if para_dict['frac_train'] is None:
        return dataset, df
    else:
        splitter = dc.splits.RandomSplitter(dataset_file)
        train_loader, test_loader = splitter.train_test_split(dataset, 
                                                              seed=para_dict['rand_seed'], 
                                                              frac_train = para_dict['frac_train'])
        df = df.set_index(id_field)
        train_df = df.loc[list(train_loader.ids)]
        test_df = df.loc[list(test_loader.ids)]
        train_df = train_df.reset_index()
        test_df = test_df.reset_index()
        return train_loader, test_loader , train_df, test_df

#----------------------------------------------
# functions for transfer value calculating (layer1/2)
def ST_model_layer1(n_features, layer_size, pretrained_params,
            lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipnorm=5):
    training_dat = Input(shape = (n_features,), dtype = 'float32')
    X = Dense(layer_size[0], activation = 'relu')(training_dat)
    model = Model(training_dat, X)
    optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
    model.compile(loss = 'mean_squared_error', optimizer=optimizer)
    
    # set parameters
    model.layers[1].set_weights(pretrained_params[0])
    return model

def ST_model_layer2(n_features, layer_size,pretrained_params,
            lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipnorm=5):
    training_dat = Input(shape = (n_features,), dtype = 'float32')
    X = Dense(layer_size[0], activation = 'relu')(training_dat)
    X = Dense(layer_size[1], activation = 'relu')(X)
    model = Model(training_dat, X)
    optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
    model.compile(loss = 'mean_squared_error', optimizer=optimizer)
    
    # set parameters
    model.layers[1].set_weights(pretrained_params[0])
    model.layers[2].set_weights(pretrained_params[1])
    return model








