import torch
import itertools
import copy

from visar.VISAR_model import visar_model
from visar.dataloader.pytorch_utils import compound_FP_loader
from visar.dataloader.AttentiveFP_utils import feature_dict_loader
import os
import pandas as pd
import pdb
import numpy as np

def hyperparam_screening(model_class, para_dict, candidate_params_dict, 
                         mode = 'grid_search', epoch = 10, epoch_num = 2, model_type = 'DNN'):

    current_path = os.getcwd()
    para_dict['dataset_file'] = para_dict['dataset_file'].replace(r'.', current_path, 1)
    if model_type == 'AFP':
        para_dict['feature_file'] = para_dict['feature_file'].replace(r'.', current_path, 1)
    hp_path = os.path.join(os.getcwd(), 'logs', para_dict['model_name'] + '_HP_screen')
    if not os.path.exists(hp_path):
        os.mkdir(hp_path)
    os.chdir(hp_path)

    if mode == 'grid_search':

        # identify all possible combinations of candidate params
        keys = list(candidate_params_dict.keys())
        # count combination
        index_list = [[xx for xx in range(len(candidate_params_dict[key]))] for key in candidate_params_dict]
        index_comb = list(itertools.product(*index_list))

        param_list = []
        param_tracking = [[] for _ in range(len(keys))]
        for cnt in range(len(index_comb)):
            temp_param = copy.deepcopy(para_dict)
            for k in range(len(keys)):
                temp_param[keys[k]] = candidate_params_dict[keys[k]][index_comb[cnt][k]]
                param_tracking[k].append(candidate_params_dict[keys[k]][index_comb[cnt][k]])
            temp_param['model_name'] = temp_param['model_name'] + '_%d' % cnt
            temp_param['epoch'] = epoch
            temp_param['epoch_num'] = epoch_num
            param_list.append(temp_param)

        param_df = pd.DataFrame(param_tracking).transpose()
        param_df.columns = keys

        # initial params saving
        best_param = {}
        best_perform = -1000
        best_temp_param = ''
        test_evaluation = [[],[],[]]

        # tracking store
        for rep in range(3):
            para_dict['rand_seed'] = rep
            if model_type == 'DNN':
                train_loader, valid_loader, _, _, para_dict = compound_FP_loader(para_dict)
            elif model_type == 'AFP':
                train_loader, valid_loader, _, _, para_dict = feature_dict_loader(para_dict)
            
            for mm in range(len(param_list)):
                param = param_list[mm]
                
                if para_dict['normalize']:
                    param['mean_list'] = para_dict['mean_list']
                    param['std_list'] = para_dict['std_list']
                    param['ratio_list'] = para_dict['ratio_list']
                    
                print('-----------------------------')
                temp_model = model_class(param)
                temp_model.model_init()
                temp_model.save_path
            
                # model quick training
                temp_model.fit(train_loader, valid_loader)

                # deterimine if the best perform should be updated
                current_scores = temp_model.evaluate(valid_loader)
                current_scores = np.mean(np.array(current_scores[0]))
                test_evaluation[rep].append(current_scores)
                if current_scores > best_perform:
                    best_perform = current_scores
                    best_param = param
                    best_temp_param = param_df.iloc[mm]
                print('Current score: %.3f' % current_scores)
                print('Best score: %.3f' % best_perform)
                print('Best_param:')
                print(best_temp_param)

                # clear working directory
                os.system('rm -r %s' % temp_model.model_path)

    # save tracking performance
    
    param_df['R2 score rep1'] = test_evaluation[0]
    param_df['R2 score rep2'] = test_evaluation[1]
    param_df['R2 score rep3'] = test_evaluation[2]
    param_df.to_csv('screen_performance_tracking.csv', index = False)
    os.chdir(current_path)

    return best_param
