import deepchem as dc

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

def ST_model_hyperparam_screen(params_dict):
    '''
    hyperparameter screening using deepchem package
    input: fname --- name of the file of raw data containing chemicals and the value for each assay;
           task --- list of task names (supporting 1 or more, but must be a list)
           FP_type --- name of the fingprint type (one of 'Circular_2048', 'Circular_1024', 'Morgan', 'MACCS', 'RDKit_FP')
           params_dict --- dictionary containing the parameters along with the screening range
           log_path --- the directory saving the log file
    output: the log; log file saved in log_path
    '''
    fname = params_dict['dataset_file']
    task_names = params_dict['task_list']
    FP_type = params_dict['feature_type']
    smiles_field = params_dict['smiles_field']
    id_field = params_dict['id_field']
    log_path = './logs/'

    log_output = []
    for task in task_names:
        print('----------------------------------------------')
        dataset_file = '%s/temp.csv' % (log_path)
        dataset, _ = prepare_dataset(params_dict)
        for cnt in range(3):
            print('Preparing dataset for %s of rep %d...' % (task, cnt + 1))
            splitter = dc.splits.RandomSplitter(dataset_file)
            train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset)
    
            print('Hyperprameter screening ...')
            metric = dc.metrics.Metric(dc.metrics.r2_score)
            optimizer = dc.hyper.HyperparamOpt(ST_model_builder)
            best_dnn, best_hyperparams, all_results = optimizer.hyperparam_search(params_dict,
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
    
    return  log_output

#------------------------------------

def RobustMT_model_builder(model_params, model_dir):
    model = dc.models.RobustMultitaskRegressor(**model_params)
    return model

def RobustMT_model_hyperparam_screen(params_dict):
    '''
    hyperparameter screening using deepchem package
    input: fname --- name of the file of raw data containing chemicals and the value for each assay;
           task --- list of task names (supporting 1 or more, but must be a list)
           FP_type --- name of the fingprint type (one of 'Circular_2048', 'Circular_1024', 'Morgan', 'MACCS', 'RDKit_FP')
           params_dict --- dictionary containing the parameters along with the screening range
           log_path --- the directory saving the log file
    output: the log; log file saved in log_path
    '''
    fname = params_dict['dataset_file']
    task_names = params_dict['task_list']
    FP_type = params_dict['feature_type']
    smiles_field = params_dict['smiles_field']
    id_field = params_dict['id_field']
    log_path = './logs/'

    log_output = []
    print('----------------------------------------------')
    dataset_file = '%s/temp.csv' % (log_path)
    dataset, _ = prepare_dataset(fname, task_names, dataset_file, FP_type, 
                                 smiles_field = smiles_field, 
                                 add_features = None,
                                 id_field = id_field, model_flag = 'MT')
    for cnt in range(3):
        print('Preparing dataset of rep %d...' % ((cnt + 1)))
        splitter = dc.splits.RandomSplitter(dataset_file)
        train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset)
    
        print('Hyperprameter screening ...')
        metric = dc.metrics.Metric(dc.metrics.r2_score, np.mean)
        optimizer = dc.hyper.HyperparamOpt(RobustMT_model_builder)
        best_dnn, best_hyperparams, all_results = optimizer.hyperparam_search(params_dict,
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
    
    return  log_output





