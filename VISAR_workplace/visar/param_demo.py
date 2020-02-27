FP_dim = {'Morgan_2048': 2048,
          'Morgan_1024': 1024,
          'MACCS': 167}

para_dict_DC_MT = {
    'model_name': 'DC_MT_reg',
    'task_list': ['T107','T108'],
    'eval_type': 'regression',
    # input data related params:
    'dataset_file': './data/MT_data_clean_June28.csv',
    'feature_type': 'Circular_2048',
    'id_field': 'molregno',
    'smiles_field': 'salt_removed_smi',
    'model_flag': 'MT',
    'add_features': None,
    'frac_train': 0.9,
    'rand_seed': 0,
    # model architecture related parameters:
    'layer_sizes': [128, 64],
    'dropouts': 0.5,
    # model training related parameters:
    'learning_rate': 0.001,
    'GPU': False,
    'epoch': 40, # training epoch of each round (saving model at the end of each round)
    'epoch_num': 2, # how many rounds
    # viz file processing related parameters:
    'model_architecture':'ST',
    'valid_cutoff': None, 
    'n_layer': 2
}

para_dict_DC_robustMT = {
    'model_name': 'DC_RobustMT_reg',
    'task_list': ['T107', 'T108'],
    'eval_type': 'regression',
    # input data related params:
    'dataset_file': './data/MT_data_clean_June28.csv',
    'feature_type': 'Circular_2048',
    'id_field': 'molregno',
    'smiles_field': 'salt_removed_smi',
    'model_flag': 'MT',
    'add_features': None,
    'frac_train': 0.8,
    'rand_seed': 0,
    # model architecture related parameters:
    'layer_sizes': [128, 64],
    'bypass_layer_sizes': [64],
    'dropouts': 0.5,
    'bypass_dropouts': 0.5,
    # model training related parameters:
    'learning_rate': 0.001,
    'GPU': False,
    'epoch': 40, # training epoch of each round (saving model at the end of each round)
    'epoch_num': 2, # how many rounds
    # viz file processing related parameters:
    'model_architecture':'RobustMT',
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
    'baseline_type': 'SVR',
    # viz file processing related parameters:
    'model_architecture': 'ST'
}

para_dict_pyDNN = {
    'model_name': 'baseline_reg',
    'task_list': ['T107', 'T108'],
    'eval_type': 'regression',
    # input data related params:
    'dataset_file': './data/MT_data_clean_June28.csv',
    'feature_type': 'Morgan',
    'id_field': 'molregno',
    'smiles_field': 'salt_removed_smi',
    'model_flag': 'MT',
    'add_features': None,
    'frac_train': 0.8,
    'rand_seed': 0,
    'batch_size': 100,
    # model architecture related parameters:
    'layer_nodes': [128, 64, 2],
    'dropouts': 0.5,
    # model training related parameters:
    'learning_rate': 0.001,
    'GPU': False,
    'epoch': 40, # training epoch of each round (saving model at the end of each round)
    'epoch_num': 2, # how many rounds
    'optimizer': 'RMSprop',
    # viz file processing related parameters:
    'model_architecture': 'MT',
    'hidden_layer': 1,
}

para_dict_AttentiveFP = {
    'model_name': 'Attentive_FP_reg',
    'task_list':['T107'],
    # input data related params:
    'dataset_file': './data/MT_data_clean_June28.csv',
    'feature_file': './data/kinase_sample_data.pickle',
    'smiles_field': 'salt_removed_smi',
    'id_field': 'molregno',
    'num_atom_features': 10,
    'num_bond_features': 20,
    'model_flag': 'ST',
    'add_features': None,
    'frac_train': 0.8,
    'rand_seed': 0,
    'batch_size': 100,
    # model architecture related parameters:
    'radius': 2,
    'T': 1,
    'fingerprint_dim': 128,
    'output_units_num': 2,
    'dropouts': 0.4,
    'batch_normalization': True,
    # viz file processing related parameters:
    'model_architecture': 'MT',
    'hidden_layer': 1,  
}

