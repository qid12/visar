import os
import json 
import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.svm import LinearSVR

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import matthews_corrcoef, accuracy_score
from math import sqrt

#from visar_utils import update_bicluster
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans

import matplotlib.cm as cm
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdMolDescriptors
from IPython.display import SVG
#import cairosvg

from visar.utils.visar_utils import update_bicluster, FP_dim
import pdb
import copy

class visar_model:
    def __init__(self, para_dict, *args, **kwargs):

        self.para_dict = copy.deepcopy(para_dict)

        # set working environment for the model
        self.work_path = os.path.join(os.getcwd(), 'logs')
        if 'model_name' not in para_dict:
            self.para_dict['model_name'] = 'Model'
        self.model_path = os.path.join(self.work_path, self.para_dict['model_name'])
        self.save_path = os.path.join(self.model_path, 'model')

        if not os.path.exists(self.work_path):
            os.mkdir(self.work_path)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        if self.load_param() is None:  # save model training params
            self.save_param()

    # --------------------------------
    def model_init(self):
        if self.para_dict['baseline_type'] == 'SVR':
            self.model = LinearSVR(C = 1.0, epsilon = 0.2)
        if self.para_dict['baseline_type'] == 'RidgeCV':
            alphas = np.logspace(start = -1, stop = 2, num = 20)
            self.model = linear_model.RidgeCV(alphas)
        return

    def predict(self, data_loader):
        new_X = np.c_[[1]*data_loader.X.shape[0], data_loader.X]
        return self.model.predict(new_X).reshape(-1,1)

    def evaluate(self, data_loader):
        if self.para_dict['eval_type'] == 'regression':
            y_true = data_loader.y.flatten()
            y_pred = self.predict(data_loader).flatten()
            return sqrt(mean_squared_error(y_pred, y_true)), pearsonr(y_pred, y_true)
        elif self.para_dict['eval_type'] == 'classification':
            y_true = data_loader.y.flatten()
            y_pred = []
            # print(outputs.shape)
            # print(labels.shape)
            for a in outputs:
                if a[0]>a[1]:
                    y_pred.append(0)
                else:
                    y_pred.append(1)
            acc = accuracy_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)
            return acc, mcc

    def fit(self, data_loader):
        new_X = np.c_[[1]*data_loader.X.shape[0], data_loader.X]
        self.model.fit(new_X, data_loader.y.flatten())

    # --------------------------------
    def save_model(self):
        return
        
    def load_model(self):
        return

    # --------------------------------
    def save_param(self, path = None):
        if path==None:
            filepath = os.path.join(self.model_path, 'train_parameters.json')
        else:
            filepath = os.path.join(path, 'train_parameters.json')
        with open(filepath, 'w') as f:
            json.dump(self.para_dict, f, indent=2)

    def load_param(self, path = None):
        if path == None:
            filepath = os.path.join(self.model_path, 'train_parameters.json')
        else:
            filepath = os.path.join(path, 'train_parameters.json')
        if os.path.exists(filepath):
            return json.load(open(filepath, 'r'))
        return None

    #---------------------------------
    #@property
    #def valid_mask(self):
    #    if 'valid_cutoff' in self.para_dict and (self.para_dict['valid_cutoff'] is not None):
    #        final_merit = self.evaluate(
    #        return final_merit > valid_cutoff
    #    else:
    #        return np.array([True] * self.n_tasks)
        
    #-------------------------------------------------
    
    def get_coords(self, train_loader, custom_loader = None, mode = 'default'):
        if mode == 'default':
            N_training = train_loader.X.shape[0]
            transfer_values = train_loader.X
            if not custom_loader is None:
                N_custom = custom_loader.X.shape[0]
                transfer_values = np.concatenate((train_loader.X, custom_loader.X), axis = 0)

            pca = PCA(n_components = 20)
            value_reduced_20d = pca.fit_transform(transfer_values)
            tsne = TSNE(n_components = 2)
            value_reduced = tsne.fit_transform(value_reduced_20d)

            if not custom_loader is None:
                return value_reduced[0:N_training,:], value_reduced[N_training:(N_training+N_custom),:]
            else:
                return value_reduced, None

    def generate_compound_df(self, data_loader, df, coord_values, id_field):
        pred_mat = self.predict(data_loader)
        #pred_mat = pred_mat[:,self.valid_mask]

        if len(pred_mat.shape) == 1:
            pred_mat = pred_mat.reshape(pred_mat.shape[0],1)

        # if normalized, transform them back to original scale
        if self.para_dict['normalize']:
            for nn in range(pred_mat.shape[1]):
                pred_mat[:,nn] = pred_mat[:,nn].flatten() * self.para_dict['std_list'][nn] + self.para_dict['mean_list'][nn]
        
        pred_df = pd.DataFrame(pred_mat)
        pred_df.columns = ['pred_' + xx for xx in self.tasks]
        pred_df['chembl_id'] = data_loader.ids

        coord_df = pd.DataFrame(coord_values)
        coord_df.columns = ['x', 'y']
        coord_df['chembl_id'] = data_loader.ids
        
        if not type(df[id_field].iloc[0]) == str:
            coord_df['chembl_id'] = coord_df['chembl_id'].astype(int)
            pred_df['chembl_id'] = pred_df['chembl_id'].astype(int)
        compound_df = pd.merge(df, coord_df, left_on = id_field, right_on = 'chembl_id')
        compound_df = pd.merge(compound_df, pred_df, on = 'chembl_id')
        return compound_df

    def cluster_MiniBatch(self, values, grain_size = 30):
        # clustering using KMeans minibatch algorithm
        n_clusters = int(values.shape[0] / grain_size)
        n_clusters = min(n_clusters, 500)
        mbk = MiniBatchKMeans(init = 'k-means++', init_size = 501, n_clusters = n_clusters, batch_size=100,
                              n_init = 10, max_no_improvement=10, verbose=0, random_state=0)
        mbk.fit(values)
        return mbk

    def generate_batch_df(self, train_loader, custom_loader, coord_values1, coord_values2):
        pred_mat = self.predict(train_loader)

        if len(pred_mat.shape) == 1:
            pred_mat = pred_y.reshape(pred_mat.shape[0],1)

        values = coord_values1
        N_training = pred_mat.shape[0]

        if not coord_values2 is None:
            values = np.concatenate((coord_values1, coord_values2), axis = 0)
            pred_mat2 = self.predict(custom_loader)
            if len(pred_mat2.shape) == 1:
                pred_mat2 = pred_mat2.reshape(pred_mat2.shape[0],1)

            pred_mat = np.concatenate((pred_mat, pred_mat2), axis = 0)

        mbk = self.cluster_MiniBatch(values)

        mbk.means_labels_unique = np.unique(mbk.labels_)
        n_row = len(np.unique(mbk.labels_))
        n_col = pred_mat.shape[1]
        cluster_info_mat = np.zeros((n_row, (n_col + 3)))

        for k in range(n_row):
            mask = mbk.labels_ == mbk.means_labels_unique[k]
            cluster_info_mat[k, 0:n_col] = np.nanmean(pred_mat[mask,:], axis = 0)
            cluster_info_mat[k, n_col] = sum(mask)
            cluster_info_mat[k, (n_col + 1) : (n_col + 3)] = np.nanmean(values[mask,:], axis = 0)
        self.compound_df1['label'] = mbk.labels_[0:N_training]
        self.batch_df = pd.DataFrame(cluster_info_mat)
        self.batch_df.columns = ['avg_' + xx for xx in self.tasks] + ['size', 'coordx', 'coordy']
        self.batch_df['Label_id'] = mbk.means_labels_unique

        if not coord_values2 is None:
            self.compound_df2['label'] = mbk.labels_[N_training: pred_mat.shape[0]]
        return

    def calculate_gradients(self):
        return self.model.coef_.flatten()[1:].reshape(-1,1)

    def generate_task_df(self):
        grad_mat = self.calculate_gradients()
        self.task_df = pd.DataFrame(grad_mat)
        self.task_df.columns = self.tasks
        return

    def generate_viz_results(self, train_loader, train_df, output_prefix,
                             custom_loader = None, custom_df = None, prev_model = None):
        if not prev_model is None:
            self.load_model(self, prev_model)

        self.tasks = self.para_dict['task_list']
        
        print('------------- Prepare information for chemicals ------------------')
        coord_values1, coord_values2 = self.get_coords(train_loader, custom_loader)

        self.compound_df1 = self.generate_compound_df(train_loader, train_df, 
                                                      coord_values1, self.para_dict['id_field'])
        if not custom_loader is None:
            self.compound_df2 = self.generate_compound_df(custom_loader, custom_df, 
                                                          coord_values2, self.para_dict['custom_id_field'])
        
        print('------------- Prepare information for minibatches ------------------')
        # clustering
        self.generate_batch_df(train_loader, custom_loader, coord_values1, coord_values2)
        
        print('------------- Prepare information for tasks ------------------')
        # derivative/gradient/sensitivity calculation
        self.generate_task_df()
        
        print('------- Generate color labels with default K of 5 --------')
        # color mapping
        batch_df, task_df, compound_df = update_bicluster(self.batch_df, self.task_df, self.compound_df1, 
                                                          mode = 'ST', K = 5)
        if not custom_loader is None:
            lut2 = dict(zip(batch_df['Label_id'], batch_df['batch_label_color']))
            lut22 = dict(zip(batch_df['Label_id'], batch_df['batch_label']))
            lut222 = dict(zip(compound_df['label'], compound_df['label_color']))
            self.compound_df2['batch_label_color'] = self.compound_df2['label'].map(lut2)
            self.compound_df2['batch_label'] = self.compound_df2['label'].map(lut22)
            self.compound_df2['label_color'] = self.compound_df2['label'].map(lut222)

        print('-------------- Saving datasets ----------------')
        # replace smiles field to 'canonical_smiles'
        switch_field = lambda item:'canonical_smiles' if  item == self.para_dict['smiles_field'] else item
        compound_df.columns = [switch_field(item) for item in compound_df.columns.tolist()]
        
        compound_df.to_csv('{}/{}_compound_df.csv'.format(self.model_path, output_prefix), index = False)
        batch_df.to_csv('{}/{}_batch_df.csv'.format(self.model_path, output_prefix), index = False)
        task_df.to_csv('{}/{}_task_df.csv'.format(self.model_path, output_prefix), index = False)

        if not custom_loader is None:
            switch_field = lambda item:'canonical_smiles' if item == self.para_dict['custom_smiles_field'] else item
            self.compound_df2.columns = [switch_field(item) for item in self.compound_df2.columns.tolist()]
            self.compound_df2.to_csv('{}/{}_compound_custom_df.csv'.format(self.model_path, output_prefix), index = False)
        
        return

    def generate_instance_analysis(self, smiles_string,
                                   pos_cut = 3, neg_cut = -3, nBits = 2048):
        """
        Notice task_df must be generated before running this function
        map the gradient of Morgan fingerprint bit on the molecule
        Input:
            smi - the smiles of the molecule (a string)
            gradient - the 2048 coeffients of the feature
            cutoff - if positive, get the pos where the integrated weight is bigger than the cutoff;
                    if negative, get the pos where the integrated weight is smaller than the cutoff
        Output:
            two list of atom ids (positive and negative)   
        """
        #self.generate_task_df(train_loader, prev_model, valid_mask)
        gradient_dict = {}
        
        # generate mol 
        mol = Chem.MolFromSmiles(smiles_string)
        highlit_pos_dict = {}
        highlit_neg_dict = {}
        atomsToUse_dict = {}
        for item in self.task_df.columns.tolist():
            gradient_dict[item] = np.array(self.task_df[item].values)
            gradient = gradient_dict[item]
            # get the bit info of the Morgan fingerprint
            bi = {}
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius = 2, bitInfo=bi, nBits=nBits)
            onbits = list(fp.GetOnBits())
            # calculate the integrated weight
            atomsToUse = np.zeros((len(mol.GetAtoms()),1))
            for bitId in onbits:
                atomID, radius = bi[bitId][0]
                temp_atomsToUse = []
                if radius > 0:
                    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atomID)
                    for b in env:
                        temp_atomsToUse.append(mol.GetBondWithIdx(b).GetBeginAtomIdx())
                        temp_atomsToUse.append(mol.GetBondWithIdx(b).GetEndAtomIdx())
                else:
                    temp_atomsToUse.append(atomID)
                    env = None
                temp_atomsToUse = list(set(temp_atomsToUse))
                atomsToUse[temp_atomsToUse] += gradient[bitId]
            # get the postively/negatively contributed atom ids
            highlit_pos = []
            highlit_neg = []
            for i in range(len(atomsToUse)):
                if  atomsToUse[i] > pos_cut:
                    highlit_pos.append(i)
                elif atomsToUse[i] < neg_cut:
                    highlit_neg.append(i)
            highlit_neg_dict[item] = highlit_neg
            highlit_pos_dict[item] = highlit_pos

            # min max normalization
            atomsToUse = (atomsToUse - np.min(atomsToUse)) / (np.max(atomsToUse) - np.min(atomsToUse))
            atomsToUse_dict[item] = atomsToUse

        return mol, highlit_pos_dict, highlit_neg_dict, atomsToUse_dict

    def __repr__(self):
        print(self.model)

    def __str__(self):
        print(self.model)
        return








