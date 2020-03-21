import os
import json
import warnings

import deepchem as dc
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans


warnings.filterwarnings("ignore")
from visar.dataloader.deepchem_utils import (
    prepare_dataset,
    ST_model_layer1,
    ST_model_layer2
    )

from visar.VISAR_model import visar_model
from visar.utils.visar_utils import update_bicluster, FP_dim
import pdb

import tensorflow.keras.backend as K
warnings.filterwarnings("ignore")

class deepchem_regressor(visar_model):
    def __init__(self, para_dict, *args, **kwargs):
        super().__init__(para_dict, *args, **kwargs)

        # set data related parameters
        self.id_field = self.para_dict['id_field']

        # extract model_related parameters
        self.model_flag = self.para_dict['model_architecture']
        if self.para_dict['add_features'] is None:
            self.n_tasks = len(self.para_dict['task_list'])
        else:
            self.n_tasks = len(self.para_dict['task_list']) + len(self.para_dict['add_features'])
        self.n_features = FP_dim[self.para_dict['feature_type']]

        self.dropout = self.para_dict['dropouts']
        self.layer_sizes = self.para_dict['layer_sizes']

        # get training params
        self.lr = self.para_dict['learning_rate']
        self.epoch_num = self.para_dict['epoch_num']
        self.epoch = self.para_dict['epoch']

    def model_init(self):
        self.model = dc.models.MultitaskRegressor(n_tasks = self.n_tasks, 
                                               n_features = self.n_features, 
                                               layer_sizes = self.layer_sizes,
                                               dropouts = self.dropout, 
                                               learning_rate = self.lr)
        self.model.save_file = self.save_path + '/model'
        self.model.model_dir = self.save_path
        return

    def predict(self, data_loader):
        return self.model.predict(data_loader).squeeze()

    def evaluate(self, data_loader):
        metric = dc.metrics.Metric(
                dc.metrics.r2_score, np.mean, mode = 'regression')
        scores = self.model.evaluate(data_loader, [metric], [], per_task_metrics=metric)
        return scores
        

    def fit(self, train_loader, test_loader, restore_flag = False):

        train_evaluation = []
        test_evaluation = []
        for iteration in range(self.epoch_num):
            self.model.fit(train_loader, nb_epoch = self.epoch, restore = restore_flag,
                           max_checkpoints_to_keep = 1, checkpoint_interval=20)
            print('======== Iteration %d ======' % iteration)
            print("Evaluating model")
            train_scores = self.evaluate(train_loader)
            train_evaluation.append(train_scores[1]["mean-r2_score"])
            print("Training R2 score: %f" % train_scores[0]["mean-r2_score"])
            test_scores = self.evaluate(test_loader)
            test_evaluation.append(test_scores[1]["mean-r2_score"])
            print("Test R2 score: %f" % test_scores[0]["mean-r2_score"])
        
            # save evaluation scores
            train_df = pd.DataFrame(np.array(train_evaluation))
            test_df = pd.DataFrame(np.array(test_evaluation))
            train_df.columns = train_loader.get_task_names()
            test_df.columns = train_loader.get_task_names()
            train_df.to_csv(self.save_path + '/' + self.para_dict['model_name'] + '_train_log.csv', index = None)
            test_df.to_csv(self.save_path + '/' + self.para_dict['model_name'] + '_test_log.csv', index = None)

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

    def load_model(self, prev_model):
        self.model.restore(checkpoint = prev_model)
        return

    #----------------------------------
    def get_coords(self, transfer_model, train_loader, custom_loader = None, mode = 'default'):
        if mode == 'default':
            transfer_values = transfer_model.predict(train_loader.X)
            N_training = train_loader.X.shape[0]
            if not custom_loader is None:
                transfer_values2 = transfer_model.predict(custom_loader.X)
                N_custom = len(custom_loader.X)
                transfer_values = np.concatenate((transfer_values, transfer_values2), axis = 0)

            pca = PCA(n_components = 20)
            value_reduced_20d = pca.fit_transform(transfer_values)
            tsne = TSNE(n_components = 2)
            value_reduced = tsne.fit_transform(value_reduced_20d)

            if not custom_loader is None:
                return value_reduced[0:N_training,:], value_reduced[N_training:(N_training+N_custom),:]
            else:
                return value_reduced, None

    #----------------------------------
    # --------------------------------
    def get_weights_MT(self, layer_variables):
        with self.model._get_tf("Graph").as_default():
            w1 = self.model.session.run(layer_variables[0])
            b1 = self.model.session.run(layer_variables[1])
        return [w1, b1]

    def get_transfer_model(self, n_layer = 2):
        # load previous parameters
        tot_layer_variables = self.model.get_variables()
        param1 = self.get_weights_MT([tot_layer_variables[0], tot_layer_variables[1]])
        param2 = self.get_weights_MT([tot_layer_variables[2], tot_layer_variables[3]])
    
        n_features = param1[0].shape[0]
        layer_size = [param1[0].shape[1], param2[0].shape[1]]
    
        if n_layer == 1:
            transfer_model = ST_model_layer1(self.n_features, layer_size, [param1, param2])
        elif n_layer == 2:
            transfer_model = ST_model_layer2(self.n_features, layer_size, [param1, param2])
        else:
            print('invalid layer size!')
        return transfer_model

    # gradient calculation
    def calculate_gradients(self, X_train, task_tensor_name, prev_model):
        '''
        Calculate the gradients for each chemical
        input: X_train --- fingerprint matrix of the chemicals of interest
               prev_model -- trained neural network model
        output: the gradient matrix
        '''
        feed_dict = {}

        with tf.Graph().as_default():
            with tf.Session() as sess:
                K.set_session(sess)

                new_saver = tf.train.import_meta_graph(prev_model + '.meta')
                new_saver.restore(sess, prev_model)
                graph = tf.get_default_graph()

                feed_dict['Feature_7/PlaceholderWithDefault:0'] = X_train
                #feed_dict['Dense_7/Dense_7/Relu:0'] = X_train[0:10,0:512]
                feed_dict['Placeholder:0'] = 1.0

                op_tensor = graph.get_tensor_by_name(task_tensor_name)
                X = graph.get_tensor_by_name('Feature_7/PlaceholderWithDefault:0')
                #X = graph.get_tensor_by_name('Dense_7/Dense_7/Relu:0')

                reconstruct = tf.gradients(op_tensor, X)[0]
                out = sess.run(reconstruct, feed_dict = feed_dict)[0]

        K.clear_session()
        return out

    def generate_task_df(self, dataset, prev_model):
        SHARE_LAYER = 'Dense_2/Dense_2/BiasAdd:0'
        
        grad_mat = self.calculate_gradients(dataset.X, SHARE_LAYER, prev_model).reshape(-1,1)
        self.task_df = pd.DataFrame(grad_mat)
        self.task_df.columns = ['SHARE']

        return

    #-------------------------------

    def generate_viz_results(self, train_loader, train_df, output_prefix,
                             custom_loader = None, custom_df = None, prev_model = None):
        self.load_model(prev_model)

        # get the actual task list from log files
        test_log_df = pd.read_csv(self.save_path + '/' + self.para_dict['model_name'] + '_test_log.csv')
        self.tasks = test_log_df.columns.values

        print('------------- Prepare information for chemicals ------------------')
        # calculate transfer values and coordinates
        model_transfer = self.get_transfer_model(n_layer = self.para_dict['n_layer'])
        coord_values1, coord_values2 = self.get_coords(model_transfer, train_loader, custom_loader)

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
        self.generate_task_df(train_loader, prev_model)

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

#-------------------------------------
#-------------------------------------

class deepchem_robust_regressor(deepchem_regressor):
    def __init__(self, para_dict, *args, **kwargs):
        super().__init__(para_dict, *args, **kwargs)

        # set default parameters
        
        self.layer_sizes = self.para_dict['layer_sizes']
        self.bypass_layer_sizes = self.para_dict['bypass_layer_sizes']
        self.dropout = self.para_dict['dropouts']
        self.bypass_dropouts = self.para_dict['bypass_dropouts']
        self.n_bypass = len(self.layer_sizes)

    def model_init(self):
        self.model = dc.models.RobustMultitaskRegressor(n_tasks = self.n_tasks, 
                                n_features = self.n_features, layer_sizes = self.layer_sizes,
                                               bypass_layer_sizes=self.bypass_layer_sizes, 
                                               bypass_dropouts = self.bypass_dropouts,
                                               dropouts = self.dropout, learning_rate = self.lr)
        self.model.save_file = self.save_path + '/model'
        self.model.model_dir = self.save_path
        return

    # --------------------------------
    # gradient calculation
    def calculate_gradients(self, X_train, task_tensor_name, prev_model):
        '''
        Calculate the gradients for each chemical
        input: X_train --- fingerprint matrix of the chemicals of interest
               prev_model -- trained neural network model
        output: the gradient matrix
        '''
        feed_dict = {}

        with tf.Graph().as_default():
            with tf.Session() as sess:
                K.set_session(sess)

                new_saver = tf.train.import_meta_graph(prev_model + '.meta')
                new_saver.restore(sess, prev_model)
                graph = tf.get_default_graph()

                feed_dict['Feature_8/PlaceholderWithDefault:0'] = X_train
                #feed_dict['Dense_7/Dense_7/Relu:0'] = X_train[0:10,0:512]
                feed_dict['Placeholder:0'] = 1.0

                op_tensor = graph.get_tensor_by_name(task_tensor_name)
                X = graph.get_tensor_by_name('Feature_8/PlaceholderWithDefault:0')
                #X = graph.get_tensor_by_name('Dense_7/Dense_7/Relu:0')

                reconstruct = tf.gradients(op_tensor, X)[0]
                out = sess.run(reconstruct, feed_dict = feed_dict)[0]

        K.clear_session()
        return out

    def generate_task_df(self, dataset, prev_model):
        n_bypass = len(self.bypass_layer_sizes)
        TASK_LAYERS = ['Dense_%d/Dense_%d/Relu:0' % (10 + self.n_bypass * 2 * idx, 10 + self.n_bypass * 2 * idx)
                        for idx in range(self.n_tasks)]
        TASK_LAYERS = list(np.array(TASK_LAYERS))
        SHARE_LAYER = 'Dense_7/Dense_7/Relu:0'
        grad_mat = np.zeros((len(TASK_LAYERS)+1, self.n_features))

        for i in range(len(TASK_LAYERS)):
            grad_mat[i,:] = self.calculate_gradients(dataset.X, TASK_LAYERS[i], prev_model)
        grad_mat[len(TASK_LAYERS),:] = self.calculate_gradients(dataset.X, SHARE_LAYER, prev_model)
        self.task_df = pd.DataFrame(grad_mat.T)
        self.task_df.columns = list(self.tasks) + ['SHARE']

        return
        
#-------------------------------------
#-------------------------------------


