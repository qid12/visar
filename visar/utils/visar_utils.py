import pandas as pd
import numpy as np
from sklearn.cluster.bicluster import SpectralCoclustering
from sklearn import preprocessing
from bokeh.palettes import Category20_20, Category20b_20
from rdkit import Chem
from rdkit.Chem import PandasTools
import matplotlib.cm as cm
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdMolDescriptors, AllChem
import os


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

#-------------------------------------
def df2sdf(df, output_sdf_name, 
           smiles_field = 'canonical_smiles', id_field = 'chembl_id', 
           selected_batch = None):
    '''
    pack pd.DataFrame to sdf_file
    '''
    if not selected_batch is None:
        df = df.loc[df['label'] == selected_batch]
    PandasTools.AddMoleculeColumnToFrame(df,smiles_field,'ROMol')
    PandasTools.WriteSDF(df, output_sdf_name, idName=id_field, properties=df.columns)

    return

#--------------------------------------
def gradient_based_atom_contribution(smiles_string, lig_pdb_fname, task_df):
    raw_mol = Chem.MolFromPDBFile(lig_pdb_fname)
    template = Chem.MolFromSmiles(smiles_string)
    mol = AllChem.AssignBondOrdersFromTemplate(template, raw_mol)
    
    atomsToUse_dict = {}
    gradient_dict = {}
    for item in task_df.columns.tolist():
        gradient_dict[item] = np.array(task_df[item].values)
        gradient = gradient_dict[item]
        # get the bit info of the Morgan fingerprint
        bi = {}
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius = 2, bitInfo=bi, nBits=nBits)
        onbits = list(fp.GetOnBits())
        # calculate the integrated weight
        atomsToUse = np.zeros((len(mol.GetAtoms()),1))
        for bitId in onbits:
            atomID, radius = bi[bitId][0]
            temp_atomsToUse = []
            if radius > 0:
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atomID)
                for b in env:
                    temp_atomsToUse.append(mol.GetBondWithIdx(b).GetBeginAtomIdx())
                    temp_atomsToUse.append(mol.GetBondWithIdx(b).GetEndAtomIdx())
            else:
                temp_atomsToUse.append(atomID)
                env = None
            temp_atomsToUse = list(set(temp_atomsToUse))
            atomsToUse[temp_atomsToUse] += gradient[bitId]

        # min max normalization
        atomsToUse = (atomsToUse - np.min(atomsToUse)) / (np.max(atomsToUse) - np.min(atomsToUse))
        atomsToUse_dict[item] = atomsToUse
    return mol, atomsToUse_dict

#-----------------------------------
""" contribution from Hans de Winter """
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.SaltRemover import SaltRemover

#----------------------------------------------------
def _InitialiseNeutralisationReactions():
    patts= (
        # Imidazoles
        ('[n+;H]','n'),
        # Amines
        ('[N+;!H0]','N'),
        # Carboxylic acids and alcohols
        ('[$([O-]);!$([O-][#7])]','O'),
        # Thiols
        ('[S-;X1]','S'),
        # Sulfonamides
        ('[$([N-;X2]S(=O)=O)]','N'),
        # Enamines
        ('[$([N-;X2][C,N]=C)]','N'),
        # Tetrazoles
        ('[n-]','[nH]'),
        # Sulfoxides
        ('[$([S-]=O)]','S'),
        # Amides
        ('[$([N-]C=O)]','N'),
        )
    return [(Chem.MolFromSmarts(x),Chem.MolFromSmiles(y,False)) for x,y in patts]

_reactions=None
def NeutraliseCharges_RemoveSalt(smiles, reactions=None):
    global _reactions
    if reactions is None:
        if _reactions is None:
            _reactions=_InitialiseNeutralisationReactions()
        reactions=_reactions
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        remover = SaltRemover()
        mol, deleted = remover.StripMolWithDeleted(mol)
        replaced = False
        for i,(reactant, product) in enumerate(reactions):
            while mol.HasSubstructMatch(reactant):
                replaced = True
                rms = AllChem.ReplaceSubstructs(mol, reactant, product)
                mol = rms[0]
        if replaced:
            return (Chem.MolToSmiles(mol,True), True)
        else:
            return (smiles, False)
    else:
        return (None, False)



