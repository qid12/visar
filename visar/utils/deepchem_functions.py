import deepchem as dc
import pdb
import numpy as np
import pandas as pd

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from visar.dataloader.deepchem_utils import prepare_dataset

#------------------------------------------------------

def ST_model_builder(model_params, model_dir):
    model = dc.models.MultitaskRegressor(**model_params)
    return model

def ST_model_hyperparam_screen(params_dict, candidate_params_dict):
    '''
    hyperparameter screening using deepchem package
    input: fname --- name of the file of raw data containing chemicals and the value for each assay;
           task --- list of task names (supporting 1 or more, but must be a list)
           FP_type --- name of the fingprint type (one of 'Circular_2048', 'Circular_1024', 'Morgan', 'MACCS', 'RDKit_FP')
           params_dict --- dictionary containing the parameters along with the screening range
           log_path --- the directory saving the log file
    output: the log; log file saved in log_path
    '''
    current_path = os.getcwd()
    params_dict['dataset_file'] = params_dict['dataset_file'].replace(r'.', current_path, 1)
    hp_path = os.path.join(os.getcwd(), 'logs', params_dict['model_name'] + '_HP_screen')
    if not os.path.exists(hp_path):
        os.mkdir(hp_path)
    os.chdir(hp_path)
    log_path = hp_path

    log_output = []
    for task in task_names:
        print('----------------------------------------------')
        dataset_file = '%s/temp.csv' % (log_path)
        
        for cnt in range(3):
            print('Preparing dataset for %s of rep %d...' % (task, cnt + 1))
            train_dataset, valid_dataset, _, _, _ = prepare_dataset(params_dict)
    
            print('Hyperprameter screening ...')
            metric = dc.metrics.Metric(dc.metrics.r2_score)
            optimizer = dc.hyper.HyperparamOpt(ST_model_builder)
            best_dnn, best_hyperparams, all_results = optimizer.hyperparam_search(candidate_params_dict,
                                                                      train_dataset,
                                                                      valid_dataset, [],
                                                                      metric)
            # get the layer size and dropout rate of all_results
            for (key, value) in all_results.items():
                log_output.append('rep%d\t%s\t%s\t%s' % (cnt, task, str(key), str(value)))
    
            print('Generate performace report ...')
            with open('%s/hyperparam_log.txt' % (log_path), 'w') as f:
                for line in log_output:
                    f.write("%s\n" % line)
        os.system('rm %s' % dataset_file)
    os.chdir(current_path)
    
    return  log_output

#------------------------------------

def RobustMT_model_builder(model_params, model_dir):
    model = dc.models.RobustMultitaskRegressor(**model_params)
    return model

def RobustMT_model_hyperparam_screen(params_dict, candidate_params_dict):
    '''
    hyperparameter screening using deepchem package
    input: fname --- name of the file of raw data containing chemicals and the value for each assay;
           task --- list of task names (supporting 1 or more, but must be a list)
           FP_type --- name of the fingprint type (one of 'Circular_2048', 'Circular_1024', 'Morgan', 'MACCS', 'RDKit_FP')
           params_dict --- dictionary containing the parameters along with the screening range
           log_path --- the directory saving the log file
    output: the log; log file saved in log_path
    '''
    current_path = os.getcwd()
    params_dict['dataset_file'] = params_dict['dataset_file'].replace(r'.', current_path, 1)
    hp_path = os.path.join(os.getcwd(), 'logs', params_dict['model_name'] + '_HP_screen')
    if not os.path.exists(hp_path):
        os.mkdir(hp_path)
    os.chdir(hp_path)
    log_path = hp_path

    log_output = []

    for cnt in range(3):
        print('----------------------------------------------')
        dataset_file = '%s/temp.csv' % (log_path)
        print('Preparing dataset of rep %d...' % ((cnt + 1)))
        train_dataset, valid_dataset, _, _, _ = prepare_dataset(params_dict)
    
        print('Hyperprameter screening ...')
        metric = dc.metrics.Metric(dc.metrics.r2_score, np.mean)
        optimizer = dc.hyper.HyperparamOpt(RobustMT_model_builder)
        best_dnn, best_hyperparams, all_results = optimizer.hyperparam_search(candidate_params_dict,
                                                                      train_dataset,
                                                                      valid_dataset, [],
                                                                      metric)
        # get the layer size and dropout rate of all_results
        for (key, value) in all_results.items():
            log_output.append('rep%d\t%s\t%s' % (cnt, str(key), str(value)))
    
        print('Generate performace report ...')
        with open('%s/%s_hyperparam_log.txt' % (log_path, params_dict['model_name']), 'w') as f:
            for line in log_output:
                f.write("%s\n" % line)
        os.system('rm %s' % dataset_file)
    os.chdir(current_path)
    
    return  log_output





