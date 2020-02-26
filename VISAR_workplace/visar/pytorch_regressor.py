import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error
from visar.visar_utils import FP_dim, update_bicluster 
from visar.pytorch_models import DNN_regressor

#================================================

class pytorch_DNN_model(visar):
    def __init__(self, para_dict, *args, **kwargs):
        super().__init__(self, para_dict, *args, **kwargs)

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
        self.layer_sizes = self.para_dict['layer_sizes']
        self.n_layers = len(self.layer_sizes)

        # get training params
        self.lr = self.para_dict['learning_rate']
        self.epoch_num = self.para_dict['epoch_num']
        self.epoch = self.para_dict['epoch']


    def model_init(self):
        self.model = DNN_regressor(self.n_features, self.layer_sizes, self.dropout,
                                   epoch = self.epoch_num, lr = self.lr)
    
    #---------------------    
    def fit(self, data_loader):
        self.train()
        optimizer = self.optimizers()

        for e in range(saved_epoch, self.epoch):
            total_loss = 0
            outputs_train = []

            for features, labels, mask, _ in data_loader:
                logps = self.forward(features)
                loss = self.loss_func(logps, y, mask)
                total_loss += loss

                if self.GPU:
                    outputs_train.append(logs.cpu().detach().numpy())
                else:
                    outputs_train.append(logps.detach().numpy())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('Epoch: %d Loss=%.3f' % (e+1, total_loss))

        
        values = np.concatenate([i for _, i, _, _ in data_loader])
        masks = np.concatenate([i for _, _, i, _ in data_loader])
        self.evaluate(np.concatenate(outputs_train), values, masks)
    
        def loss_func(self, output, target, mask):
        pdb.set_trace()
        if self.GPU:
            target = target.cuda()
            mask = mask.cuda()
        out = (output[mask] - target[mask])**2
        loss = out.mean()

        return loss

    def optimizers(self):
        if self.optimizer == 'Adam':
            return optim.Adam(self.parameters(), lr=self.lr)

        elif self.optimizer == 'RMSprop':
            return optim.RMSprop(self.parameters(), lr=self.lr)

        elif self.optimizer == 'SGD':
            return optim.SGD(self.parameters(), lr=self.lr)
    #-----------------------------

    def fit(self, train_loader, test_loader):

        train_evaluation = []
        test_evaluation = []
        for iteration in range(self.epoch_num):
            self.model.fit(train_loader)

            print('======== Iteration %d ======' % iteration)
            print("Evaluating model")
            train_scores = self.evaluate(train_loader)
            train_evaluation.append(train_scores[0])
            print("Training R2 score: %.3f" % np.mean(np.array(train_scores[0])))
            test_scores = self.evaluate(test_loader)
            test_evaluation.append(test_scores[0])
            print("Test R2 score: %.3f" % np.mean(np.array(test_scores[0])))
        
            # save evaluation scores
            train_df = pd.DataFrame(np.array(train_evaluation))
            test_df = pd.DataFrame(np.array(test_evaluation))
            train_df.columns = self.tasks
            test_df.columns = self.tasks
            train_df.to_csv(self.save_path + '/train_log.csv', index = None)
            test_df.to_csv(self.save_path + '/test_log.csv', index = None)

            # save model
            self.save_model('Epoch_' + str(e+1), self.data_dict())


    def evaluate(self, data_loader):
        outputs = []
        labels = []
        weights = []
        for X, y, w, _ in data_loader:
            outputs.append(self.model.predict(X))
            labels.append(y)
            weights.append(w)

        Xs = np.concatenate(outputs, axis = 0)
        ys = np.concatenate(labels, axis = 0)
        ws = np.concatenate(w, axis = 0)
        return self.model.evaluate(Xs, ys, ws)

    def predict(self, data_loader):
        Xs = []
        for X, _, _, _ in data_loader:
            Xs.append(X)
        Xs = np.concatenate(Xs, axis = 0)
        return self.model.predict(Xs)

    def save_model(self, filename):
        torch.save(self.model, os.path.join(self.save_path, filename))

    def load_model(self):

        for e in range(self.epoch, 0, -1):
            if os.path.isfile(os.path.join(self.save_path, 'Epoch_' + str(e))):
                # print(os.path.join(self.save_path, 'Epoch_' + str(e)))
                self.load_state_dict(torch.load(os.path.join(self.save_path, 'Epoch_' + str(e))))
                return e
        return 0

    #----------------------------------
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
        for _, _, _, ids, _ in data_loader:
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

