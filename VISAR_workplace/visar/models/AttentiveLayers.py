import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
from sklearn.metrics import r2_score, mean_squared_error
import pdb

class Fingerprint(nn.Module):

    def __init__(self, radius, T, input_feature_dim, input_bond_dim,\
            fingerprint_dim, output_units_num, p_dropout, 
            batch_normalization = False, momentum = 0.5,GPU = False):
        super(Fingerprint, self).__init__()
        self.GPU = GPU
        self.do_bn = batch_normalization
        self.bns = []

        # graph attention for atom embedding
        atom_fc = nn.Linear(input_feature_dim, fingerprint_dim)
        setattr(self, 'fc0', atom_fc)
        #self._set_init(atom_fc)
        self.atom_fc = atom_fc
        if self.do_bn:
            bn = nn.BatchNorm1d(fingerprint_dim, momentum = momentum)
            setattr(self, 'bn0', bn)
            self.bns.append(bn)

            bn = nn.BatchNorm2d(fingerprint_dim, momentum = momentum)
            setattr(self, 'bn1', bn)
            self.bns.append(bn)
            
            bn = nn.BatchNorm1d(fingerprint_dim, momentum = momentum)
            setattr(self, 'bn2', bn)
            self.bns.append(bn)
            
            
        self.neighbor_fc = nn.Linear(input_feature_dim+input_bond_dim, fingerprint_dim)
        self.GRUCell = nn.ModuleList([nn.GRUCell(fingerprint_dim, fingerprint_dim) for r in range(radius)])
        self.align = nn.ModuleList([nn.Linear(2*fingerprint_dim,1) for r in range(radius)])
        self.attend = nn.ModuleList([nn.Linear(fingerprint_dim, fingerprint_dim) for r in range(radius)])
        # graph attention for molecule embedding
        self.mol_GRUCell = nn.GRUCell(fingerprint_dim, fingerprint_dim)
        self.mol_align = nn.Linear(2*fingerprint_dim,1)
        self.mol_attend = nn.Linear(fingerprint_dim, fingerprint_dim)
        # you may alternatively assign a different set of parameter in each attentive layer for molecule embedding like in atom embedding process.
#         self.mol_GRUCell = nn.ModuleList([nn.GRUCell(fingerprint_dim, fingerprint_dim) for t in range(T)])
#         self.mol_align = nn.ModuleList([nn.Linear(2*fingerprint_dim,1) for t in range(T)])
#         self.mol_attend = nn.ModuleList([nn.Linear(fingerprint_dim, fingerprint_dim) for t in range(T)])
        
        self.dropout = nn.Dropout(p=p_dropout)
        self.output = nn.Linear(fingerprint_dim, output_units_num)

        self.radius = radius
        self.T = T

    def _set_init(self, layer):
        init.normal_(layer.weight, mean=0., std=.1) # could be changed here!
        init.constant_(layer.bias, 0)

    def forward(self, atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask):
        #layer_input = []
        #pre_activation = []
        
        atom_mask = atom_mask.unsqueeze(2)
        batch_size,mol_length,num_atom_feat = atom_list.size()
        atom_feature_preact = self.atom_fc(atom_list)
        
        #pre_activation.append(atom_feature_preact)
        if self.do_bn: atom_feature_preact = self.bns[0](atom_feature_preact.transpose(1,2)).transpose(1,2) # transpose of the dataset
        atom_feature = F.leaky_relu(atom_feature_preact)
        #layer_input.append(atom_feature)
        bond_neighbor = [bond_list[i][bond_degree_list[i]] for i in range(batch_size)]
        bond_neighbor = torch.stack(bond_neighbor, dim=0)
        atom_neighbor = [atom_list[i][atom_degree_list[i]] for i in range(batch_size)]
        atom_neighbor = torch.stack(atom_neighbor, dim=0)
        
        # then concatenate them
        neighbor_feature = torch.cat([atom_neighbor, bond_neighbor],dim=-1)
        neighbor_feature_preact = self.neighbor_fc(neighbor_feature)
        #pre_activation.append(neighbor_feature_preact)
        if self.do_bn: neighbor_feature_preact = self.bns[1](neighbor_feature_preact.transpose(1,3)).transpose(1,3) # transpose of the dataset
        neighbor_feature = F.leaky_relu(neighbor_feature_preact)
        #layer_input.append(neighbor_feature)

        # generate mask to eliminate the influence of blank atoms
        attend_mask = atom_degree_list.clone()
        attend_mask[attend_mask != mol_length-1] = 1
        attend_mask[attend_mask == mol_length-1] = 0
        attend_mask = attend_mask.unsqueeze(-1)
        #attend_mask = attend_mask.type(torch.cuda.FloatTensor).unsqueeze(-1)

        softmax_mask = atom_degree_list.clone()
        softmax_mask[softmax_mask != mol_length-1] = 0
        softmax_mask[softmax_mask == mol_length-1] = -9e8 # make the softmax value extremly small
        softmax_mask = softmax_mask.unsqueeze(-1)
        #softmax_mask = softmax_mask.type(torch.cuda.FloatTensor).unsqueeze(-1)

        batch_size, mol_length, max_neighbor_num, fingerprint_dim = neighbor_feature.shape
        atom_feature_expand = atom_feature.unsqueeze(-2).expand(batch_size, mol_length, max_neighbor_num, fingerprint_dim)
        feature_align = torch.cat([atom_feature_expand, neighbor_feature],dim=-1)
        
        align_score = F.leaky_relu(self.align[0](self.dropout(feature_align)))
#             print(attention_weight)
        align_score = align_score + softmax_mask
        attention_weight = F.softmax(align_score,-2)
#             print(attention_weight)
        attention_weight = attention_weight * attend_mask
#         print(attention_weight)
        neighbor_feature_transform = self.attend[0](self.dropout(neighbor_feature))
#             print(features_neighbor_transform.shape)
        context = torch.sum(torch.mul(attention_weight,neighbor_feature_transform),-2)
#             print(context.shape)
        context = F.elu(context)
        context_reshape = context.view(batch_size*mol_length, fingerprint_dim)
        atom_feature_reshape = atom_feature.view(batch_size*mol_length, fingerprint_dim)
        atom_feature_reshape = self.GRUCell[0](context_reshape, atom_feature_reshape)
        atom_feature = atom_feature_reshape.view(batch_size, mol_length, fingerprint_dim)

        #do nonlinearity
        activated_features = F.relu(atom_feature)

        for d in range(self.radius-1):
            # bonds_indexed = [bond_list[i][torch.cuda.LongTensor(bond_degree_list)[i]] for i in range(batch_size)]
            neighbor_feature = [activated_features[i][atom_degree_list[i]] for i in range(batch_size)]
            
            # neighbor_feature is a list of 3D tensor, so we need to stack them into a 4D tensor first
            neighbor_feature = torch.stack(neighbor_feature, dim=0)
            atom_feature_expand = activated_features.unsqueeze(-2).expand(batch_size, mol_length, max_neighbor_num, fingerprint_dim)

            feature_align = torch.cat([atom_feature_expand, neighbor_feature],dim=-1)

            align_score = F.leaky_relu(self.align[d+1](self.dropout(feature_align)))
    #             print(attention_weight)
            align_score = align_score + softmax_mask
            attention_weight = F.softmax(align_score,-2)
#             print(attention_weight)
            attention_weight = attention_weight * attend_mask
#             print(attention_weight)
            neighbor_feature_transform = self.attend[d+1](self.dropout(neighbor_feature))
    #             print(features_neighbor_transform.shape)
            context = torch.sum(torch.mul(attention_weight,neighbor_feature_transform),-2)
    #             print(context.shape)
            context = F.elu(context)
            #layer_input.append(context)
            context_reshape = context.view(batch_size*mol_length, fingerprint_dim)
#             atom_feature_reshape = atom_feature.view(batch_size*mol_length, fingerprint_dim)
            atom_feature_reshape = self.GRUCell[d+1](context_reshape, atom_feature_reshape)
            atom_feature = atom_feature_reshape.view(batch_size, mol_length, fingerprint_dim)
            
            # do nonlinearity
            activated_features = F.relu(atom_feature)
        
        mol_feature = torch.sum(activated_features * atom_mask, dim=-2)
        
        # do nonlinearity
        #pdb.set_trace()
        #pre_activation.append(mol_feature)
        if self.do_bn: mol_feature = self.bns[2](mol_feature) # transpose of the dataset
        activated_features_mol = F.relu(mol_feature)
        #layer_input.append(activated_features_mol)
        
        mol_softmax_mask = atom_mask.clone()
        mol_softmax_mask[mol_softmax_mask == 0] = -9e8
        mol_softmax_mask[mol_softmax_mask == 1] = 0
        #mol_softmax_mask = mol_softmax_mask.type(torch.cuda.FloatTensor)
        
        for t in range(self.T):
            
            mol_prediction_expand = activated_features_mol.unsqueeze(-2).expand(batch_size, mol_length, fingerprint_dim)
            mol_align = torch.cat([mol_prediction_expand, activated_features], dim=-1)
            mol_align_score = F.leaky_relu(self.mol_align(mol_align))
            mol_align_score = mol_align_score + mol_softmax_mask
            mol_attention_weight = F.softmax(mol_align_score,-2)
            mol_attention_weight = mol_attention_weight * atom_mask
#             print(mol_attention_weight.shape,mol_attention_weight)
            activated_features_transform = self.mol_attend(self.dropout(activated_features))
#             aggregate embeddings of atoms in a molecule
            mol_context = torch.sum(torch.mul(mol_attention_weight,activated_features_transform),-2)
#             print(mol_context.shape,mol_context)
            mol_context = F.elu(mol_context)
            mol_feature = self.mol_GRUCell(mol_context, mol_feature)
#             print(mol_feature.shape,mol_feature)

            # do nonlinearity
            activated_features_mol = F.relu(mol_feature)           
            
        #mol_prediction = F.relu(self.output(self.dropout(mol_feature)))
        mol_prediction = self.output(self.dropout(mol_feature))
        
        return atom_feature, mol_prediction

    def forward4viz(self, atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask):     

        atom_mask = atom_mask.unsqueeze(2)
        batch_size,mol_length,num_atom_feat = atom_list.size()
        atom_feature_preact = self.atom_fc(atom_list)        
        if self.do_bn: atom_feature_preact = self.bns[0](atom_feature_preact.transpose(1,2)).transpose(1,2) # transpose of the dataset
        atom_feature = F.leaky_relu(atom_feature_preact)

        atom_feature_viz = []
        atom_feature_viz.append(self.atom_fc(atom_list))

        bond_neighbor = [bond_list[i][bond_degree_list[i]] for i in range(batch_size)]
        bond_neighbor = torch.stack(bond_neighbor, dim=0)
        atom_neighbor = [atom_list[i][atom_degree_list[i]] for i in range(batch_size)]
        atom_neighbor = torch.stack(atom_neighbor, dim=0)

        #then catenate them
        neighbor_feature = torch.cat([atom_neighbor, bond_neighbor],dim=-1)
        neighbor_feature_preact = self.neighbor_fc(neighbor_feature)
        if self.do_bn: neighbor_feature_preact = self.bns[1](neighbor_feature_preact.transpose(1,3)).transpose(1,3) # transpose of the dataset
        neighbor_feature = F.leaky_relu(neighbor_feature_preact)

        # generate mask to eliminate the influence of blank atoms
        attend_mask = atom_degree_list.clone()
        attend_mask[attend_mask != mol_length-1] = 1
        attend_mask[attend_mask == mol_length-1] = 0
        attend_mask = attend_mask.unsqueeze(-1)
        #attend_mask = attend_mask.type(torch.cuda.FloatTensor).unsqueeze(-1)

        softmax_mask = atom_degree_list.clone()
        softmax_mask[softmax_mask != mol_length-1] = 0
        softmax_mask[softmax_mask == mol_length-1] = -9e8 # make the softmax value extremly small
        softmax_mask = softmax_mask.unsqueeze(-1)
        #softmax_mask = softmax_mask.type(torch.cuda.FloatTensor).unsqueeze(-1)

        batch_size, mol_length, max_neighbor_num, fingerprint_dim = neighbor_feature.shape
        atom_feature_expand = atom_feature.unsqueeze(-2).expand(batch_size, mol_length, max_neighbor_num, fingerprint_dim)
        feature_attention = torch.cat([atom_feature_expand, neighbor_feature],dim=-1)
        
        align_score = self.dropout(F.leaky_relu(self.align[0](feature_attention)))
#             print(attention_weight)
        align_score = align_score + softmax_mask
        attention_weight = F.softmax(align_score,-2)
#             print(attention_weight)
        attention_weight = attention_weight * attend_mask
#         print(attention_weight)
        atom_attention_weight_viz = []
        atom_attention_weight_viz.append(attention_weight)
        
        neighbor_feature_transform = self.attend[0](self.dropout(neighbor_feature))
#             print(features_neighbor_transform.shape)
        context = torch.sum(torch.mul(attention_weight,neighbor_feature_transform),-2)
#             print(context.shape)
        context = F.elu(context)
        context_reshape = context.view(batch_size*mol_length, fingerprint_dim)
        atom_feature_reshape = atom_feature.view(batch_size*mol_length, fingerprint_dim)
        atom_feature_reshape = self.GRUCell[0](context_reshape, atom_feature_reshape)
        atom_feature = atom_feature_reshape.view(batch_size, mol_length, fingerprint_dim)


        #do nonlinearity
        activated_features = F.relu(atom_feature)
        atom_feature_viz.append(activated_features)

        for d in range(self.radius-1):
            # bonds_indexed = [bond_list[i][torch.cuda.LongTensor(bond_degree_list)[i]] for i in range(batch_size)]
            neighbor_feature = [activated_features[i][atom_degree_list[i]] for i in range(batch_size)]
            
            # neighbor_feature is a list of 3D tensor, so we need to stack them into a 4D tensor first
            neighbor_feature = torch.stack(neighbor_feature, dim=0)
            atom_feature_expand = activated_features.unsqueeze(-2).expand(batch_size, mol_length, max_neighbor_num, fingerprint_dim)

            feature_attention = torch.cat([atom_feature_expand, neighbor_feature],dim=-1)

            align_score = self.dropout(F.leaky_relu(self.align[d+1](feature_attention)))
    #             print(attention_weight)
            align_score = align_score + softmax_mask
            attention_weight = F.softmax(align_score,-2)
#             print(attention_weight)
            attention_weight = attention_weight * attend_mask
            atom_attention_weight_viz.append(attention_weight)
#             print(attention_weight)
            neighbor_feature_transform = self.attend[d+1](self.dropout(neighbor_feature))
    #             print(features_neighbor_transform.shape)
            context = torch.sum(torch.mul(attention_weight,neighbor_feature_transform),-2)
    #             print(context.shape)
            context = F.elu(context)
            context_reshape = context.view(batch_size*mol_length, fingerprint_dim)
#             atom_feature_reshape = atom_feature.view(batch_size*mol_length, fingerprint_dim)
            atom_feature_reshape = self.GRUCell[d+1](context_reshape, atom_feature_reshape)
            atom_feature = atom_feature_reshape.view(batch_size, mol_length, fingerprint_dim)
            
            # do nonlinearity
            activated_features = F.relu(atom_feature)
            atom_feature_viz.append(activated_features)

        # when the descriptor value are unbounded, like partial charge or LogP
        mol_feature_unbounded_viz = []
        mol_feature_unbounded_viz.append(torch.sum(atom_feature * atom_mask, dim=-2)) 
        
        mol_feature = torch.sum(activated_features * atom_mask, dim=-2)
        if self.do_bn: mol_feature = self.bns[2](mol_feature) # transpose of the dataset
        activated_features_mol = F.relu(mol_feature)
        
        # when the descriptor value has lower or upper bounds
        mol_feature_viz = []
        mol_feature_viz.append(mol_feature) 
        
        mol_attention_weight_viz = []
        mol_softmax_mask = atom_mask.clone()
        mol_softmax_mask[mol_softmax_mask == 0] = -9e8
        mol_softmax_mask[mol_softmax_mask == 1] = 0
        #mol_softmax_mask = mol_softmax_mask.type(torch.cuda.FloatTensor)
        
        for t in range(self.T):
            
            mol_prediction_expand = activated_features_mol.unsqueeze(-2).expand(batch_size, mol_length, fingerprint_dim)
            mol_align = torch.cat([mol_prediction_expand, activated_features], dim=-1)
            mol_align_score = self.dropout(F.leaky_relu(self.mol_align(mol_align)))
            mol_align_score = mol_align_score + mol_softmax_mask
            mol_attention_weight = F.softmax(mol_align_score,-2)
            mol_attention_weight = mol_attention_weight * atom_mask
#             print(mol_attention_weight.shape,mol_attention_weight)
            mol_attention_weight_viz.append(mol_attention_weight)

            activated_features_transform = self.mol_attend(self.dropout(activated_features))
            mol_context = torch.sum(torch.mul(mol_attention_weight,activated_features_transform),-2)
#             print(mol_context.shape,mol_context)
            mol_context = F.elu(mol_context)
            mol_feature = self.mol_GRUCell(mol_context, mol_feature)
#             print(mol_feature.shape,mol_feature)

            mol_feature_unbounded_viz.append(mol_feature)
            #do nonlinearity
            activated_features_mol = F.relu(mol_feature)           
            mol_feature_viz.append(activated_features_mol)
            
        mol_prediction = self.output(self.dropout(mol_feature))
            
        return atom_feature_viz, atom_attention_weight_viz, mol_feature_viz, mol_feature_unbounded_viz, mol_attention_weight_viz, mol_prediction

    def get_transfer_values(self, x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, n_layer = 0):
        _, _, mol_feature_viz, _, _, _ = self.forward4viz(x_atom, x_bonds, x_atom_index, x_bond_index, x_mask)
        return mol_feature_viz[n_layer].detach().numpy()

    def get_attention_values(self, x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, n_layer = 0):
        _, _, _, _, mol_attention_weight_viz, _ = self.forward4viz(x_atom, x_bonds, x_atom_index, x_bond_index, x_mask)
        return mol_attention_weight_viz[n_layer].squeeze().detach().numpy()

    def predict(self, x_atom, x_bonds, x_atom_index, x_bond_index, x_mask):
        self.eval()
        with torch.no_grad():
            atom_feature, mol_prediction = self.forward(x_atom, x_bonds, x_atom_index, x_bond_index, x_mask)
        return atom_feature.numpy(), mol_prediction

    def evaluate(self, outputs, values, mask):
        outputs = torch.FloatTensor(outputs)
            
        if len(values.shape) == 1 or values.shape[1] == 1:
            y_pred = outputs.flatten()[mask]
            y_true = values.flatten()[mask]
            
            if self.GPU:
                y_pred = y_pred.cpu().numpy()
            r2 = r2_score(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)

            return r2, mse
            
        else:  # multiple assays
            r2_store = []
            mse_store = []
            for i in range(values.shape[1]):
                y_pred = outputs[mask[:,i],i].flatten()
                y_true = values[mask[:,i],i].flatten()
                if self.GPU:
                    y_pred = y_pred.cpu().numpy()
                r2_store.append(r2_score(y_true, y_pred))
                mse_store.append(mean_squared_error(y_true, y_pred))

            print(r2_store)
            print(mse_store)

            return r2_store, mse_store




