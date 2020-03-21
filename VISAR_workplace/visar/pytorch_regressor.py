import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error
from visar.utils.visar_utils import FP_dim, update_bicluster 
from visar.models.pytorch_models import DNNx2_regressor
from visar.VISAR_model import visar_model
import os
import pandas as pd
import pdb
import numpy as np

#from visar_utils import update_bicluster
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans

from visar.models.AttentiveLayers import Fingerprint

#================================================

class pytorch_DNN_model(visar_model):
    def __init__(self, para_dict, *args, **kwargs):
        super().__init__(para_dict, *args, **kwargs)

        # set default parameters
        if 'dropouts' not in para_dict:
            self.para_dict['dropouts'] = 0.5
        if 'layer_sizes' not in para_dict:
            self.para_dict['layer_sizes'] = [128, 64, 1]

        # set data related parameters
        self.id_field = self.para_dict['id_field'] 

        # extract model_related parameters
        self.model_flag = self.para_dict['model_architecture']
        if self.para_dict['add_features'] is None:
            self.n_tasks = len(self.para_dict['task_list'])
            self.tasks = self.para_dict['task_list']
        else:
            self.n_tasks = len(self.para_dict['task_list']) + len(self.para_dict['add_features'])
            self.tasks = self.para_dict['task_list']

        self.n_features = FP_dim[self.para_dict['feature_type']]
        self.dropout = self.para_dict['dropouts']

        # get training params
        self.lr = self.para_dict['learning_rate']
        self.epoch_num = self.para_dict['epoch_num']
        self.epoch = self.para_dict['epoch']
        self.GPU = self.para_dict['GPU']
        self.optimizer = self.para_dict['optimizer']


    def model_init(self):
        self.model = DNNx2_regressor(self.n_features, self.para_dict['layer_nodes'], self.dropout, self.GPU)

    def forward(self, X_feature):
        return self.model.forward(X_feature)
    
    def loss_func(self, output, target, mask):
        if self.GPU:
            target = target.cuda()
            mask = mask.cuda()
        if self.para_dict['normalize'] and output.shape[1] > 1:
            # assemble ratio dict
            loss = 0
            for nn in range(output.shape[1]):
                if mask[:,nn].sum() > 0:
                    se = (output[mask[:,nn],nn] - target[mask[:,nn],nn])**2 
                    loss += se.mean() * self.para_dict['ratio_list'][nn] **2
        else:
            out = (output[mask] - target[mask])**2
            loss = out.mean()

        return loss

    def optimizers(self):
        if self.optimizer == 'Adam':
            return optim.Adam(self.model.parameters(), lr=self.lr)

        elif self.optimizer == 'RMSprop':
            return optim.RMSprop(self.model.parameters(), lr=self.lr)

        elif self.optimizer == 'SGD':
            return optim.SGD(self.model.parameters(), lr=self.lr)
    #-----------------------------

    def fit(self, train_loader, test_loader):

        optimizer = self.optimizers()
        train_evaluation = []
        test_evaluation = []
        saved_epoch = self.load_model()

        for iteration in range(saved_epoch, self.epoch_num):
            # training each epoch    

            for e in range(self.epoch):
                total_loss = 0
                #outputs_train = []

                for features, labels, mask, _ in train_loader:
                    logps = self.forward(features)
                    loss = self.loss_func(logps, labels, mask)
                    total_loss += loss

                    #if self.GPU:
                    #    outputs_train.append(logps.cpu().detach().numpy())
                    #else:
                    #    outputs_train.append(logps.detach().numpy())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if e % 10 == 0:
                    print('Epoch: %d Loss=%.3f' % (e+1, total_loss))       

            print('======== Iteration %d ======' % iteration)
            print("Evaluating model")
            train_scores = self.evaluate(train_loader)
            train_evaluation.append(train_scores[0])
            print("Training R2 score: %.3f" % np.mean(np.array(train_scores[0])))
            print("Training MSE score: %.3f" % np.mean(np.array(train_scores[1])))
            test_scores = self.evaluate(test_loader)
            test_evaluation.append(test_scores[0])
            print("Test R2 score: %.3f" % np.mean(np.array(test_scores[0])))
            print("Test MSE score: %.3f" % np.mean(np.array(test_scores[1])))
            print('============================')
        
            # save evaluation scores
            train_df = pd.DataFrame(np.array(train_evaluation))
            test_df = pd.DataFrame(np.array(test_evaluation))
            train_df.columns = self.tasks
            test_df.columns = self.tasks
            train_df.to_csv(self.save_path + '/' + self.para_dict['model_name'] + '_train_log.csv', index = None)
            test_df.to_csv(self.save_path + '/' + self.para_dict['model_name'] + '_test_log.csv', index = None)

            # save model
            self.save_model('Epoch_' + str(iteration + 1), self.model.state_dict())


    def evaluate(self, data_loader):
        outputs = []
        labels = []
        weights = []
        for X, y, w, _ in data_loader:
            if self.GPU:
                outputs.append(self.forward(X).cpu().detach().numpy())
            else:
                outputs.append(self.forward(X).detach().numpy())
            labels.append(y)
            weights.append(w)

        Xs = np.concatenate(outputs, axis = 0)
        ys = np.concatenate(labels, axis = 0)
        ws = np.concatenate(weights, axis = 0)
        return self.model.evaluate(Xs, ys, ws)

    def predict(self, data_loader):
        outputs = []
        for X, _, _, _ in data_loader:
            if self.GPU:
                outputs.append(self.forward(X).cpu().detach().numpy())
            else:
                outputs.append(self.forward(X).detach().numpy())
        Xs = np.concatenate(outputs, axis = 0)
        return Xs

    def save_model(self, filename, model):
        torch.save(model, os.path.join(self.save_path, filename))

    def load_model(self):

        for e in range(self.epoch, 0, -1):
            if os.path.isfile(os.path.join(self.save_path, 'Epoch_' + str(e))):
                # print(os.path.join(self.save_path, 'Epoch_' + str(e)))
                self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'Epoch_' + str(e))))
                return e
        return 0

    #----------------------------------
    def get_coords(self, n_layer, train_loader, custom_loader = None, mode = 'default'):
        if mode == 'default':
            transfer_values = []
            for Xs, _, _, _ in train_loader:
                transfer_values.append(self.model.get_transfer_values(Xs, n_layer))
            transfer_values = np.concatenate(transfer_values, axis = 0)
            N_training = transfer_values.shape[0]
            
            if not custom_loader is None:
                transfer_values2 = []
                for Xs, _, _, _ in custom_loader:
                    transfer_values2.append(self.model.get_transfer_values(Xs, n_layer))
                transfer_values2 = np.concatenate(transfer_values2, axis = 0)

                N_custom = transfer_values2.shape[0]
                transfer_values = np.concatenate((transfer_values, transfer_values2), axis = 0)

            pca = PCA(n_components = 20)
            value_reduced_20d = pca.fit_transform(transfer_values)
            tsne = TSNE(n_components = 2)
            value_reduced = tsne.fit_transform(value_reduced_20d)

            if not custom_loader is None:
                return value_reduced[0:N_training,:], value_reduced[N_training:(N_training+N_custom),:]
            else:
                return value_reduced, None

    def generate_compound_df(self, data_loader, df, coord_values, id_field):
        data_loader_ids = []
        for _, _, _, ids in data_loader:
            data_loader_ids += ids
        
        pred_mat = self.predict(data_loader)
        
        # if normalized, transform them back to original scale
        if self.para_dict['normalize']:
            for nn in range(pred_mat.shape[1]):
                pred_mat[:,nn] = pred_mat[:,nn].flatten() * self.para_dict['std_list'][nn] + self.para_dict['mean_list'][nn]
        
        #pred_mat = pred_mat[:,self.valid_mask]
        pred_df = pd.DataFrame(pred_mat)
        pred_df.columns = ['pred_' + xx for xx in self.tasks]
        pred_df['chembl_id'] = data_loader_ids

        coord_df = pd.DataFrame(coord_values)
        coord_df.columns = ['x', 'y']
        coord_df['chembl_id'] = data_loader_ids
        
        if not type(df[id_field].iloc[0]) == str:
            coord_df['chembl_id'] = coord_df['chembl_id'].astype(int)
            pred_df['chembl_id'] = pred_df['chembl_id'].astype(int)
        compound_df = pd.merge(df, coord_df, left_on = id_field, right_on = 'chembl_id')
        compound_df = pd.merge(compound_df, pred_df, on = 'chembl_id')
        return compound_df

    def generate_task_df(self, data_loader, df):        
        Xs = np.concatenate([i for i, _, _, _ in data_loader])
        masks = np.concatenate([i for _, _, i, _ in data_loader])
        grad_mat = np.zeros((self.n_features, Xs.shape[0], len(self.tasks) + 1))
        for task_i in range(len(self.tasks)):
            grad_mat[:,:,task_i] = self.model.get_gradient_task(Xs, masks, task_i).transpose()
        grad_mat[:,:,len(self.tasks)] = self.model.get_gradient_task(Xs, masks, None).transpose()

        self.task_df = pd.DataFrame(np.mean(grad_mat,axis = 1))
        self.task_df.columns = self.tasks + ['SHARE']
        #self.task_df = grad_mat
        return

    # --------------------------------

    def generate_viz_results(self, train_loader, train_df, output_prefix,
                             custom_loader = None, custom_df = None, prev_model = None):
        self.model_init()
        self.load_model()

        # get the actual task list from log files
        test_log_df = pd.read_csv(self.save_path + '/' + self.para_dict['model_name'] + '_test_log.csv')
        self.tasks = list(test_log_df.columns.values)

        print('------------- Prepare information for chemicals ------------------')
        # calculate transfer values and coordinates
        coord_values1, coord_values2 = self.get_coords(self.para_dict['hidden_layer'], train_loader, custom_loader)

        # prediction for the training set
        self.compound_df1 = self.generate_compound_df(train_loader, train_df, 
                                                      coord_values1, 
                                                      self.para_dict['id_field'])
        if not custom_loader is None:
            self.compound_df2 = self.generate_compound_df(custom_loader, custom_df, 
                                                          coord_values2, 
                                                          self.para_dict['custom_id_field'])

        print('------------- Prepare information for minibatches ------------------')
        # clustering
        self.generate_batch_df(train_loader, custom_loader, coord_values1, coord_values2)

        print('------------- Prepare information for tasks ------------------')
        # derivative/gradient/sensitivity calculation
        self.generate_task_df(train_loader, train_df)

        print('------- Generate color labels with default K of 5 --------')
        # color mapping
        batch_df, task_df, compound_df = update_bicluster(self.batch_df, self.task_df, self.compound_df1, mode = self.model_flag, K = 5)
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
        
        # saving results
        compound_df.to_csv('{}/{}_compound_df.csv'.format(self.model_path, output_prefix), index = False)
        batch_df.to_csv('{}/{}_batch_df.csv'.format(self.model_path, output_prefix), index = False)
        task_df.to_csv('{}/{}_task_df.csv'.format(self.model_path, output_prefix), index = False)

        if not custom_loader is None:
            switch_field = lambda item:'canonical_smiles' if item == self.para_dict['custom_smiles_field'] else item
            self.compound_df2.columns = [switch_field(item) for item in self.compound_df2.columns.tolist()]
            self.compound_df2.to_csv('{}/{}_compound_custom_df.csv'.format(self.model_path, output_prefix), index = False)
        
        return

##------------------------------------

from visar.utils.getFeatures import (
    save_smiles_dicts,
    get_smiles_array)
import pickle

class pytorch_AFP_model(pytorch_DNN_model):
    def __init__(self, para_dict, *args, **kwargs):
        super().__init__(para_dict, *args, **kwargs)

        self.num_atom_features = self.para_dict['num_atom_features']
        self.num_bond_features = self.para_dict['num_bond_features']

        self.radius = self.para_dict['radius']
        self.T = self.para_dict['T']
        self.fingerprint_dim = self.para_dict['fingerprint_dim']
        self.output_units_num = self.para_dict['output_units_num']

    def model_init(self):
        self.model = Fingerprint(self.radius, self.T, 
                                 self.num_atom_features, self.num_bond_features,
                                 self.fingerprint_dim, self.output_units_num, 
                                 self.dropout, self.para_dict['batch_normalization'], 
                                 momentum = 0.5, GPU = self.GPU)

    def forward(self, X_feature):
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask = X_feature
        return self.model.forward(x_atom, x_bonds, x_atom_index, x_bond_index, x_mask)
    
    def get_coords(self, n_layer, train_loader, custom_loader = None, mode = 'default'):
        if mode == 'default':
            transfer_values = []
            for Xs, _, _, _ in train_loader:
                x_atom, x_bonds, x_atom_index, x_bond_index, x_mask = Xs
                transfer_values.append(self.model.get_transfer_values(x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, n_layer))
            transfer_values = np.concatenate(transfer_values, axis = 0)
            N_training = transfer_values.shape[0]
            
            if not custom_loader is None:
                transfer_values2 = []
                for Xs, _, _, _ in custom_loader:
                    x_atom, x_bonds, x_atom_index, x_bond_index, x_mask = Xs
                    transfer_values2.append(self.model.get_transfer_values(x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, n_layer))
                transfer_values2 = np.concatenate(transfer_values2, axis = 0)

                N_custom = transfer_values2.shape[0]
                transfer_values = np.concatenate((transfer_values, transfer_values2), axis = 0)

            pca = PCA(n_components = 20)
            value_reduced_20d = pca.fit_transform(transfer_values)
            tsne = TSNE(n_components = 2)
            value_reduced = tsne.fit_transform(value_reduced_20d)

            if not custom_loader is None:
                return value_reduced[0:N_training,:], value_reduced[N_training:(N_training+N_custom),:]
            else:
                return value_reduced, None

    def generate_task_df(self, data_loader, df):
        feature_dicts = pickle.load(open(self.para_dict['feature_file'], 'rb'))
        smiles_list = df[self.para_dict['smiles_field']].values
        ids_to_smiles = dict(zip(df[self.para_dict['id_field']].values, 
                                 df[self.para_dict['smiles_field']].values))
        _, _, _, _, _, smiles_to_rdkit_list = get_smiles_array(smiles_list, feature_dicts)
        grad_mat = np.zeros((len(df), self.para_dict['mol_length']))
        cnt = 0
        for X_feature, _, mask, ids in data_loader:
            x_atom, x_bonds, x_atom_index, x_bond_index, x_mask = X_feature
            grad = self.model.get_attention_values(x_atom, x_bonds, x_atom_index, x_bond_index, x_mask)
            for m in range(len(ids)):
                out_weight = []
                ind_mask = x_mask[m]
                ind_weight = grad[m]
                for j, one_or_zero in enumerate(list(ind_mask)):
                    if one_or_zero == 1.0:
                        out_weight.append(ind_weight[j])
                
                ind_atom = smiles_to_rdkit_list[ids_to_smiles[ids[m]]]
                grad_mat[cnt, 0:len(ind_atom)] = np.array([out_weight[m] for m in np.argsort(ind_atom)]).flatten()
                cnt += 1
                
        self.task_df = pd.DataFrame(grad_mat)
        #self.task_df.columns = ['SHARE']

        return
    
    def generate_instance_analysis(self, smiles, grad_df, idx,
                                   pos_cut = 3, neg_cut = -3):
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
        
        # generate mol 
        mol = Chem.MolFromSmiles(smiles_string)
        values = grad_df.iloc[idx]
        atomsToUse = np.array([item for item in values if ~(item == 0.0)])
        highlit_pos = []
        highlit_neg = []
        for i in range(len(atomsToUse)):
            if  atomsToUse[i] > pos_cut:
                highlit_pos.append(i)
            elif atomsToUse[i] < neg_cut:
                highlit_neg.append(i)
        highlit_neg_dict[item] = highlit_neg
        highlit_pos_dict[item] = highlit_pos
        atomsToUse_dict[item] = atomsToUse

        return mol, highlit_pos_dict, highlit_neg_dict, atomsToUse_dict
    