import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error
from visar.visar_utils import FP_dim, update_bicluster 
from visar.AttentiveLayers import Fingerprint

#================================================

class pytorch_AFP_model(pytorch_DNN_model):
    def __init__(self, para_dict, *args, **kwargs):
        super().__init__(self, para_dict, *args, **kwargs)

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
                                 self.dropout, self.batch_normalization, momentum = 0.5)

    def evaluate(self, data_loader):
        outputs = []
        labels = []
        weights = []
        for x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, y, w, ids in data_loader:
            outputs.append(self.model.predict(x_atom, x_bonds, x_atom_index, x_bond_index, x_mask))
            labels.append(y)
            weights.append(w)
        y_pred = np.concatenate(outputs, axis = 0)
        y_true = np.concatenate(labels, axis = 0)
        ws = np.concatenate(w, axis = 0)
        return self.model.evaluate(y_pred, y_true, ws)

    def predict(self, data_loader):
    	x_atom = np.concatenate([i for i, _, _, _, _, _, _, _ in data_loader])
    	x_bonds = np.concatenate([i for _, i, _, _, _, _, _, _ in data_loader])
    	x_atom_index = np.concatenate([i for _, _, i, _, _, _, _, _ in data_loader])
    	x_bond_index = np.concatenate([i for _, _, _, i, _, _, _, _ in data_loader])
    	x_mask = np.concatenate([i for _, _, _, _, i, _, _, _ in data_loader])
    	return self.model.predict(x_atom, x_bonds, x_atom_index, x_bond_index, x_mask)

#---------------------------------------------
	def get_coords(self, n_layer, train_loader, custom_loader = None, mode = 'default'):
		if mode == 'default':
            transfer_values = []
            for Xs, _, _, _, _ in train_loader:
                transfer_values.append(self.model.get_transfer_values(Xs, n_layer))
            transfer_values = np.concatenate(transfer_values, axis = 0)
            
            N_training = transfer_values.shape[0]
            if not custom_loader is None:
                transfer_values2 = []
                for Xs, _, _, _, _ in train_loader:
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
        for _, _, _, ids, _ in data_loader:  ##!!!
            data_loader_ids += ids

        pred_mat = self.predict(Xs)
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

    def generate_task_df(self, dataset):        
        grad_mat = np.zeros((n_features, len(self.tasks) + 1))
        Xs = np.concatenate([i for i, _, _, _ in data_loader])
        masks = np.concatenate([i for _, _, i, _ in data_loader])
        for task_i in range(len(self.tasks)):
            grad_mat[:,task_i] = self.model.get_gradient_task(self, Xs, masks, task_i).reshape(-1,1)
        grad_mat[:, len(self.tasks)] = self.model.get_gradient_task(self, Xs, masks, None).reshape(-1,1)

        self.task_df = pd.DataFrame(grad_mat)
        self.task_df.columns = self.tasks + ['SHARE']

        return

    # --------------------------------

    def generate_viz_results(self, train_loader, train_df, output_prefix,
                             custom_loader = None, custom_df = None, prev_model = None):
        self.model_init()
        self.load_model()

        # get the actual task list from log files
        test_log_df = pd.read_csv(self.save_path + '/test_log.csv')
        self.tasks = test_log_df.columns.values

        print('------------- Prepare information for chemicals ------------------')
        # calculate transfer values and coordinates
        coord_values1, coord_values2 = self.get_coords(self.para_dict['n_layer'], train_loader, custom_loader)

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
        self.generate_task_df(train_loader)

        print('------- Generate color labels with default K of 5 --------')
        # color mapping
        batch_df, task_df, compound_df = update_bicluster(self.batch_df, self.task_df, self.compound_df1, mode = self.model_flag, K = 5)
        if not custom_loader is None:
            lut2 = dict(zip(batch_df['Label_id'], batch_df['batch_label_color']))
            lut22 = dict(zip(batch_df['Label_id'], batch_df['batch_label']))
            lut222 = dict(zip(compound_df['label'], compound_df['label_color']))
            compound_df2['batch_label_color'] = self.compound_df2['label'].map(lut2)
            compound_df2['batch_label'] = self.compound_df2['label'].map(lut22)
            compound_df2['label_color'] = self.compound_df2['label'].map(lut222)

        print('-------------- Saving datasets ----------------')
        # saving results
        compound_df.to_csv('{}/{}_compound_df.csv'.format(self.model_path, output_prefix), index = False)
        batch_df.to_csv('{}/{}_batch_df.csv'.format(self.model_path, output_prefix), index = False)
        task_df.to_csv('{}/{}_task_df.csv'.format(self.model_path, output_prefix), index = False)

        if not custom_loader is None:
            compound_df2.to_csv(output_prefix + 'compound_custom_df.csv', index = False)
        
        return







