import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error

class DNN_regressor(nn.Module):

    def __init__(self, n_features, layer_nodes, dropouts, 
                 GPU = False, epoch = 20, lr = 0.001, optimizer = 'RMSprop'):
        super(DNN_regressor, self).__init__(n_features, layer_nodes, dropouts)

        self.GPU = GPU
        self.epoch = epoch
        self.lr = lr
        self.optimizer = optimizer

        self.n_layers = len(layer_nodes)
        self.fcs = []
        self.fcs.append(nn.Linear(in_features = n_features,
                        out_features = self.layer_nodes[0]))
        for i in range(1, self.n_layers):
            self.fcs.append(nn.Linear(in_features = self.layer_nodes[i-1],
                                      out_features = self.layer_nodes[i]))

        if self.GPU:
            self.cuda()

    def forward(self, Xs):
        batch_size = len(Xs)

        if self.GPU:
            X = Xs.cuda()
        else:
            X = torch.FloatTensor(Xs)
        
        X = torch.flatten(X, start_dim = 1)

        out = self.fcs[0](X)
        for i in range(1, self.n_layers):
            out = F.dropout(F.relu(self.fcs[i](out)), p = dropouts)

        return out

    def forward4pred(self, Xs):
        batch_size = len(Xs)

        if self.GPU:
            X = Xs.cuda()
        else:
            X = torch.FloatTensor(Xs)

        X = torch.flatten(X, start_dim = 1)

        out = self.fcs[0](X)
        for i in range(0, self.n_layers):
            out = F.relu(self.fcs[i](out))

        return out

    def forward4optim(self, Xs):
        batch_size = len(Xs)

        # turns off the gradient descent for all params
        for param in self.parameters():
            param.requires_grad = False

        if self.GPU:
            X = Xs.cuda()
        else:
            X = torch.FloatTensor(Xs)
        
        X = torch.flatten(X, start_dim = 1)
        self.X_variable = torch.autograd.Variable(X, requires_grad = True)

        out = self.fcs[0](self.X_variable)
        for i in range(1, self.n_layers):
            out = F.relu(self.fcs[i](out))

        return out

    def get_transfer_values(self, Xs, n_layer = 2):
        batch_size = len(Xs)

        if self.GPU:
            X = Xs.cuda()
        else:
            X = torch.FloatTensor(Xs)

        X = torch.flatten(X, start_dim = 1)

        for i in range(0, n_layer):
            out = F.relu(self.fcs[i])
        return out

    def predict(self, Xs):
        self.eval()
        with torch.no_grad():
            outputs = self.forward4pred(Xs)
        if self.GPU:
            return outputs.cpu().detach().numpy()
        else:
            return outputs.detach().numpy()

    def get_gradient_task(self, Xs, mask, task):
        out = self.forward4optim(Xs)
        if task is not None:
            act = out[:,task][:,mask]
        else:
            act = out[mask].sum()  ##!!!!
        act.sum().backward()

        return self.X_variable.grad

    def loss_func(self, input, target, mask):
        out = (input[mask] - target[mask])**2
        loss = out.mean()

        return loss

    def optimizers(self):
        if self.optimizer == 'Adam':
            return optim.Adam(self.parameters(), lr=self.lr)

        elif self.optimizer == 'RMSprop':
            return optim.RMSprop(self.parameters(), lr=self.lr)

        elif self.optimizer == 'SGD':
            return optim.SGD(self.parameters(), lr=self.lr)

    def fit(self, data_loader):
        self.train()
        optimizer = self.optimizers()

        for e in range(saved_epoch, self.epoch):
            total_loss = 0
            outputs_train = []

            for features, labels, mask, _ in data_loader:
                if self.GPU:
                    y = torch.cuda.FloatTensor(labels)
                else:
                    y = torch.FloatTensor(labels)

                logps = self.forward(features)
                loss = self.loss_func(logps, y, mask)
                total_loss += loss

                if self.GPU:
                    outputs_train.append(logs.cpu().detach().numpy())
                else:
                    outputs_train.append(logps.detach().numpy())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('Epoch: %d Loss=%.3f' % (e+1, total_loss))

        
        values = np.concatenate([i for _, i, _, _ in data_loader])
        masks = np.concatenate([i for _, _, i, _ in data_loader])
        self.evaluate(np.concatenate(outputs_train), values, masks)
    
    def evaluate(self, outputs, values, mask):
        if len(values.shape) == 1 or values.shape[1] == 1:
            y_pred = outputs.flatten()[mask]
            y_true = values.flatten()[mask]
            r2 = r2_score(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)

            return r2, mse
            
        else:  # multiple assays
            r2_store = []
            mse_store = []
            for i in range(values.shape[1]):
                y_pred = outputs[:,i].flatten()[mask[:,i]]
                y_true = values[:,i].flatten()[mask[:,i]]
                r2_store.append(r2_score(y_true, y_pred))
                mse_store.append(mean_squared_error(y_true, y_pred))

                print(r2_store)
                print(mse_store)

            return r2_store, mse_store   

    #-----------------------------------------



