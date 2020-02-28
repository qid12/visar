import pandas as pd
import numpy as np
from sklearn.cluster.bicluster import SpectralCoclustering
from sklearn import preprocessing
from bokeh.palettes import Category20_20, Category20b_20

FP_dim = {'Circular_2048': 2048,
          'Circular_1024': 1024,
          'Morgan': 2048,
          'MACCS': 167,
          'graph attentive fingerprint': None}

def extract_clean_dataset(subset_names, MT_df, add_features = None, id_field = 'molregno', smiles_field = 'salt_removed_smi'):
    '''
    input: subset_names --- a list of the task names
           MT_df --- pd.DataFrame of the total raw data
    output: subset of raw data， containing only the selected tasks
    '''
    extract_column = subset_names + [id_field, smiles_field]
    sub_df = MT_df[extract_column]
    
    n_tasks = len(subset_names)
    mask_mat = np.sum(np.isnan(np.matrix(MT_df[subset_names])), axis = 1).reshape(-1).flatten()
    mask_new = np.array(mask_mat < n_tasks)[0]
    
    if add_features is None:
        extract_df = sub_df.loc[mask_new]
    else:
        new_sub_df = MT_df[extract_column + add_features]
        extract_df = new_sub_df.loc[mask_new]

        # normalization!!
        all_fields = subset_names + add_features
        for i in range(len(all_fields)):
            temp_mask = ~np.isnan(np.array(extract_df[all_fields[i]].tolist()))
            extract_df[all_fields[i]].loc[temp_mask] = preprocessing.scale(extract_df[all_fields[i]].dropna())
        
    print('Extracted dataset shape: ' + str(extract_df.shape))
    return extract_df

#--------------------------------------------------------

def update_bicluster(batch_df, task_df, compound_df, mode = 'RobustMT', K = 5):
    if mode == 'RobustMT':
        n_tasks = task_df.shape[1] - 1
    elif mode == 'ST':
        n_tasks = 1
    elif mode == 'MT':
        n_tasks = task_df.shape[1]

    if not mode == 'ST':
        # cocluster of the minibatch predictive matrix
        X = preprocessing.scale(np.matrix(batch_df)[:,0:n_tasks])
        cocluster = SpectralCoclustering(n_clusters=K, random_state=0)
        cocluster.fit(X)
        batch_df['batch_label'] = cocluster.row_labels_
    else:
        rank_x = batch_df[batch_df.columns[0]].rank().tolist()
        groups = pd.qcut(rank_x, K, duplicates='drop')
        batch_df['batch_label'] = groups.codes

    # generate color hex for batch_label
    lut = dict(zip(batch_df['batch_label'].unique(), Category20_20))
    batch_df['batch_label_color'] = batch_df['batch_label'].map(lut)

    # generate color hex for compound_df
    lut2 = dict(zip(batch_df['Label_id'], batch_df['batch_label_color']))
    compound_df['batch_label_color'] = compound_df['label'].map(lut2)
    lut22 = dict(zip(batch_df['Label_id'], batch_df['batch_label']))
    compound_df['batch_label'] = compound_df['label'].map(lut22)
    groups = pd.qcut(compound_df['label'].tolist(), len(Category20b_20), duplicates='drop')
    c = [Category20b_20[xx] for xx in groups.codes]
    compound_df['label_color'] = c

    return batch_df, task_df, compound_df


