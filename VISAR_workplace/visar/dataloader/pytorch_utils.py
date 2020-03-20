import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from rdkit import Chem
from rdkit.Chem import AllChem
from visar.utils.visar_utils import extract_clean_dataset
import random
import copy

class compound_dataset(Dataset):
    def __init__(self, dataset, smiles_field, id_field, task, FP_type = 'Morgan'):
        self.dataset = dataset
        self.FP_type = FP_type
        self.task = task
        self.smiles_list = dataset[smiles_field].values
        self.id_list = dataset[id_field].values

        self.Xs = self.featurizer(dataset[smiles_field].values, FP_type)
        self.Ys = dataset[task].to_numpy()
        self.Ws = ~np.isnan(self.Ys)

    def featurizer(self, smiles_list, FP_type):
        # generate FPs
        if FP_type == 'Morgan': #2048
            mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
            fps = []
            for mol in mols:
                fps.append(np.matrix(AllChem.GetMorganFingerprintAsBitVect(mol,2)))
            fps = np.concatenate(fps)
        
        elif self.FP_type == 'MACCS': #167
            mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
            fps = []
            for mol in mols:
                fps.append(np.matrix(MACCSkeys.GenMACCSKeys(mol)))
            fps = np.asarray(np.concatenate(fps))

        else:
            print('Unsupported Fingerprint type!') 

        return fps

    def __getitem__(self, idx):
        X = np.asarray(self.Xs[idx,:]).squeeze()
        y = np.asarray(self.Ys[idx,:])
        w = np.asarray(self.Ws[idx,:])
        ids = self.id_list[idx]
        return (X, y, w, ids)

    def __len__(self):
        return len(self.dataset)

def collate_fn(data):
    X, y, w, ids = zip(*data)
    return torch.FloatTensor(X), torch.tensor(y), torch.BoolTensor(w), list(ids)


def compound_FP_loader(para_dict, max_cutoff = None):
    fname = para_dict['dataset_file']
    smiles_field = para_dict['smiles_field']
    id_field = para_dict['id_field']
    task = para_dict['task_list']
    model_flag = para_dict['model_flag']
    add_features = para_dict['add_features']
    FP_type = para_dict['feature_type']
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

    # train test partition
    #if para_dict['frac_train'] < 1:
    #    np.random.seed(para_dict['rand_seed'])
    #    msk = np.random.rand(len(df), ) < para_dict['frac_train']
    #    train_df = df[msk]
    #    test_df = df[~msk]
    if para_dict['frac_train'] < 1:
        np.random.seed(para_dict['rand_seed'])
        train_df = copy.deepcopy(df)
        test_df = copy.deepcopy(df)
        for k in range(len(task)):
            valid_index = list(df.loc[~pd.isnull(df['T11409'])].index)
            N_sample = int(np.floor(len(valid_index) * para_dict['frac_train']))
            train_index = random.sample(valid_index, N_sample)
            test_index = list(set(valid_index).difference(set(train_index)))

            train_df[task[k]].iloc[test_index] = np.nan
            test_df[task[k]].iloc[train_index] = np.nan
    else:
        train_df = df
    
    # random sample max number if too many compounds:
    if not max_cutoff is None and max_cutoff < train_df.shape[0]:
        train_df = train_df.iloc[random.sample([num for num in range(len(train_df))], max_cutoff)]

    # prepare generator
    train_loader = DataLoader(compound_dataset(train_df, smiles_field, id_field, task, FP_type), 
                              batch_size = batch_size, collate_fn = collate_fn)
    
    if para_dict['frac_train'] < 1:
        test_loader = DataLoader(compound_dataset(test_df, smiles_field, id_field, task, FP_type), 
                                  batch_size = batch_size, collate_fn = collate_fn)
        return train_loader, test_loader, train_df, test_df, para_dict
    else:
        return train_loader, train_df, para_dict


