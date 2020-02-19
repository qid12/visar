FP_dim = {'Morgan_2048': 2048,
          'Morgan_1024': 1024,
          'MACCS': 167}

para_dict_DNN = {'model_name':'DNN_2Layer_reg',
                  'hidden_layers':[128,64,1],
                  'feature_type':'Morgan_2048',
                  'GPU': False,
                  'epoch':20,
                  'learning_rate':0.001,
                  'dropout_rate':0.5,
                  'model_type': 'regression'}

para_dict_DC_robustMT = {
    'model_name': 'DC_RobustMT_reg',
    'task_list': ['T107', 'T108'],
    'eval_type': 'regression',
    # input data related params:
    'dataset_file': '../data/MT_data_clean_June28.csv',
    'feature_type': 'Morgan_2048',
    'id_field': 'molregno',
    'smiles_field': 'salt_removed_smi',
    'model_flag': 'MT',
    'add_features': None,
    'frac_train': None,
    'rand_seed': 0,
    # model architecture related parameters:
    'layer_sizes': [128, 64],
    'bypass_layer_sizes': 64,
    'dropout': 0.5,
    'bypass_dropouts': 0.5
    # model training related parameters:
    'learning_rate': 0.001,
    'GPU': False,
    'epoch': 40, # training epoch of each round (saving model at the end of each round)
    'epoch_num': 20, # how many rounds
    # viz file processing related parameters:
    'valid_cutoff': None, 
    'n_layer': 2
}

para_dict_visar = {
    'model_name': 'baseline_reg',
    'task_list': ['T107'],
    'eval_type': 'regression',
    # input data related params:
    'dataset_file': '../data/MT_data_clean_June28.csv',
    'feature_type': 'Circular_2048',
    'id_field': 'molregno',
    'smiles_field': 'salt_removed_smi',
    'model_flag': 'ST',
    'add_features': None,
    'frac_train': 0.8,
    'rand_seed': 0,
    # model architecture related parameters:
    'baseline_type': 'SVR'
}





