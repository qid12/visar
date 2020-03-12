import pandas as pd
from rdkit import Chem
from torch.utils.data import DataLoader, Dataset
import pickle

from visar.utils.getFeatures import (
    save_smiles_dicts,
    get_smiles_array)
from visar.utils.visar_utils import extract_clean_dataset
import pdb
import torch
import numpy as np

class feature_dict_dataset(Dataset):
    def __init__(self, feature_filename, df, smiles_field, id_field, tasks):
        self.feature_dict = pickle.load(open(feature_filename, 'rb'))
        self.df = df
        
        remained_df = df[df[smiles_field].isin(self.feature_dict['smiles_to_atom_mask'].keys())]
        uncovered_df = df.drop(remained_df.index)
        if len(uncovered_df) > 0:
            print('The following data is missing:')
            print(uncovered_df)
        
        self.smiles_list = remained_df[smiles_field].values
        self.tasks = tasks
        self.Ys = remained_df[tasks].to_numpy()
        self.Ws = ~np.isnan(self.Ys)
        self.ids = remained_df[id_field].values

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array([smiles], self.feature_dict)
        y = np.asarray(self.Ys[idx,:])
        ids = self.ids[idx]
        w = np.asarray(self.Ws[idx,:])
        return (x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, y, w, ids)

    def __len__(self):
        return len(self.ids)

def collate_fn(data):
    x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, y, w, ids = zip(*data)
    return [torch.Tensor(x_atom).squeeze(), torch.Tensor(x_bonds).squeeze(),torch.LongTensor(x_atom_index).squeeze(), torch.LongTensor(x_bond_index).squeeze(), torch.Tensor(x_mask).squeeze()], torch.Tensor(y).squeeze(), torch.BoolTensor(w).squeeze(), list(ids)

def feature_dict_loader(para_dict):
    fname = para_dict['dataset_file']
    feature_filename = para_dict['feature_file']
    smiles_field = para_dict['smiles_field']
    id_field = para_dict['id_field']
    task = para_dict['task_list']
    model_flag = para_dict['model_flag']
    add_features = para_dict['add_features']
    batch_size = para_dict['batch_size']
    normalize = para_dict['normalize']

    # extract clean datasets based on output_field
    MT_df = pd.read_csv(fname)

    if normalize:
        mean_list = []
        std_list = []
        mad_list = []
        ratio_list = []
        for t in task:
            mean = MT_df[t].mean(skipna = True)
            mean_list.append(mean)
            std = MT_df[t].std(skipna = True)
            std_list.append(std)
            mad = MT_df[t].mad(skipna = True)
            mad_list.append(mad)
            ratio_list.append(std/mad)
            MT_df[t] = (MT_df[t] - mean) / std
        para_dict['mean_list'] = mean_list
        para_dict['std_list'] = std_list
        para_dict['mad_list'] = mad_list
        para_dict['ratio_list'] = ratio_list

    if model_flag == 'ST':
        df = extract_clean_dataset(task, MT_df, smiles_field = smiles_field, id_field = id_field)
    elif model_flag == 'MT':
        df = extract_clean_dataset(task, MT_df, add_features = add_features, smiles_field = smiles_field, id_field = id_field)
        if not add_features is None:
            task = task + add_features

    # data preprocessing (including scale and clip, saving the related values to a json file)
    # df_new = df_new

    # train test partition
    np.random.seed(para_dict['rand_seed'])
    msk = np.random.rand(len(df), ) < para_dict['frac_train']
    train_df = df[msk]
    test_df = df[~msk]

    # prepare generator
    train_loader = DataLoader(feature_dict_dataset(feature_filename, train_df, smiles_field, id_field, task), 
                              batch_size = batch_size, collate_fn = collate_fn)
    test_loader = DataLoader(feature_dict_dataset(feature_filename, test_df, smiles_field, id_field, task), 
                              batch_size = batch_size, collate_fn = collate_fn)

    X, y, w, ids = next(iter(train_loader))
    para_dict['num_atom_features'] = X[0].shape[-1]
    para_dict['num_bond_features'] = X[1].shape[-1]

    return train_loader, test_loader, train_df, test_df, para_dict

#------------------------------------


def dataset_prepare_AttentiveFP(raw_filename, smiles_field,
                            cano_field = 'cano_smiles'):
    '''
    INPUT
        task_name: user-defined name for the training project
        raw_filename: a csv file containing smiles and task values of compounds
    '''
    feature_filename = raw_filename.replace('.csv','.pickle')
    filename = raw_filename.replace('.csv','')
    prefix_filename = raw_filename.split('/')[-1].replace('.csv','')
    output_filename = filename + '_processed.csv'
    
    print('============== Loading the raw file =====================')
    smiles_tasks_df = pd.read_csv(raw_filename)
    smilesList = smiles_tasks_df[smiles_field].values
    print("number of all smiles: ", len(smilesList))
    
    atom_num_dist = []
    remained_smiles = []
    canonical_smiles_list = []
    for smiles in smilesList:
        try:
            mol = Chem.MolFromSmiles(smiles)
            atom_num_dist.append(len(mol.GetAtoms()))
            remained_smiles.append(smiles)
            canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
        except:
            print('not successfully processed smiles: ', smiles)
            pass
    print("number of successfully processed smiles: ", len(remained_smiles))
    smiles_tasks_df = smiles_tasks_df[smiles_tasks_df[smiles_field].isin(remained_smiles)]
    smiles_tasks_df[cano_field] = canonical_smiles_list
    assert canonical_smiles_list[8] == Chem.MolToSmiles(Chem.MolFromSmiles(smiles_tasks_df[cano_field][8]), 
                                                        isomericSmiles=True)
    smiles_tasks_df.to_csv(output_filename, index = None)
    print('saving processed file as ' + output_filename)

    print('================== saving feature files ========================')
    smilesList = [smiles for smiles in canonical_smiles_list if len(Chem.MolFromSmiles(smiles).GetAtoms()) < 151]
    if os.path.isfile(feature_filename):
        print('feature file has been generated.')
    else:
        feature_dicts = save_smiles_dicts(smilesList, feature_filename)
        print('saving feature file as ', feature_filename)

    smiles_list = smilesList[0]
    x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list, feature_dicts)
    num_atom_features = x_atom.shape[-1]
    num_bond_features = x_bonds.shape[-1]
    print('Dataset {}: \n Number of atom features: {} \n Number of bond features: {}'.format(feature_filename, str(num_atom_features), str(num_bond_features)))

    return atom_num_dist, num_atom_features, num_bond_features

#------------------------------------





