import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error
import pdb

class DNNx2_regressor(nn.Module):

    def __init__(self, n_features, layer_nodes, dropouts = 0.5, GPU = False):
        super(DNNx2_regressor, self).__init__()

        self.GPU = GPU
        self.dropouts = dropouts

        self.n_layers = len(layer_nodes)
        self.layer_nodes = layer_nodes
        self.fc0 = nn.Linear(in_features = n_features,
                            out_features = self.layer_nodes[0])
        self.fc1 = nn.Linear(in_features = self.layer_nodes[0],
                             out_features = self.layer_nodes[1])
        self.fc2 = nn.Linear(in_features = self.layer_nodes[1],
                             out_features = self.layer_nodes[2])

        if self.GPU:
            self.cuda()

    def forward(self, Xs):
        batch_size = len(Xs)

        if self.GPU:
            X = Xs.cuda()
        else:
            X = Xs
        
        X = torch.flatten(X, start_dim = 1)

        out = F.dropout(F.relu(self.fc0(X)), p = self.dropouts)
        out = F.dropout(F.relu(self.fc1(out)), p = self.dropouts)
        out = self.fc2(out)

        return out

    def forward4predict(self, Xs):
        batch_size = len(Xs)

        if self.GPU:
            X = Xs.cuda()
        else:
            X = Xs

        X = torch.flatten(X, start_dim = 1)

        out = F.relu(self.fc0(X))
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
            

        return out

    def forward4viz(self, Xs):
        batch_size = len(Xs)

        # turns off the gradient descent for all params
        for param in self.parameters():
            param.requires_grad = False
        if self.GPU:
            X = torch.tensor(Xs).cuda()
        else:
            X = torch.tensor(Xs)
        
        X = torch.flatten(X, start_dim = 1)
        self.X_variable = torch.autograd.Variable(X, requires_grad = True)

        out = F.relu(self.fc0(self.X_variable))
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out

    def get_transfer_values(self, Xs, hidden_layer = 1):
        batch_size = len(Xs)

        if self.GPU:
            X = Xs.cuda()
        else:
            X = torch.FloatTensor(Xs)

        X = torch.flatten(X, start_dim = 1)
        
        out = F.relu(self.fc0(X))
        if hidden_layer == 1:
            transfer_value = out
        elif hidden_layer == 2:
            out = F.relu(self.fc1(out))
            transfer_value = out
        if self.GPU:
            transfer_value = transfer_value.detach().cpu().numpy()
        else:
            transfer_value = transfer_value.detach().numpy()
        return transfer_value

    def predict(self, Xs):
        self.eval()
        outputs = self.forward4predict(Xs)
        if self.GPU:
            return outputs.cpu().detach().numpy()
        else:
            return outputs.detach().numpy()

    def get_gradient_task(self, Xs, mask, task = None):
        out = self.forward4viz(Xs)
        if task is not None:
            act = out[mask[:,task],task]
        else:
            act = out[mask]  ##!!!!
        act.sum().backward()
        if self.GPU:
            return self.X_variable.grad.cpu().detach().numpy()
        else:
            return self.X_variable.grad.detach().numpy()
    
    def evaluate(self, outputs, values, mask):
        outputs = torch.FloatTensor(outputs)
            
        if len(values.shape) == 1 or values.shape[1] == 1:
            y_pred = outputs.flatten()[mask.flatten()]
            y_true = values.flatten()[mask.flatten()]
            
            if self.GPU:
                y_pred = y_pred.cpu().numpy()
            r2 = r2_score(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)

            return [r2], [mse]
            
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

        
        
        