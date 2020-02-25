import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from rdkit import Chem
from rdkit.Chem import AllChem
from visar.visar_utils import extract_clean_dataset

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
    return torch.tensor(X), torch.tensor(y), torch.ByteTensor(w), list(ids)


def compound_FP_loader(para_dict):
    fname = para_dict['dataset_file']
    smiles_field = para_dict['smiles_field']
    id_field = para_dict['id_field']
    task = para_dict['task_list']
    model_flag = para_dict['model_flag']
    add_features = para_dict['add_features']
    FP_type = para_dict['feature_type']
    batch_size = para_dict['batch_size']

    # extract clean datasets based on output_field
    MT_df = pd.read_csv(fname)
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
    train_loader = DataLoader(compound_dataset(train_df, smiles_field, id_field, task, FP_type), 
                              batch_size = batch_size, collate_fn = collate_fn)
    test_loader = DataLoader(compound_dataset(test_df, smiles_field, id_field, task, FP_type), 
                              batch_size = batch_size, collate_fn = collate_fn)

    return train_loader, test_loader, train_df, test_df


