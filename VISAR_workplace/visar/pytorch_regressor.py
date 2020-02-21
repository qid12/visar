import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import r2_score

class pytorch_DNN_regressor(nn.Module):
    def __init__(self, para_dict, *args, **kwargs):
        super(DNN_regressor, self).__init__(self, para_dict, *args, **kwargs)

        self.para_dict = para_dict

        # set working environment for the model
        self.work_path = os.path.join(os.getcwd(), 'logs')
        if 'model_name' not in para_dict:
            self.para_dict['model_name'] = 'Model'
        self.model_path = os.path.join(self.work_path, self.para_dict['model_name'])
        self.save_path = os.path.join(self.model_path, 'model')

        if not os.path.exsists(self.work_path):
            os.mkdir(self.work_path)
        if not os.path.exsists(self.model_path):
            os.mkdir(self.model_path)
        if not os.path.exsists(self.save_path):
            os.mkdir(self.save_path)
        if self.load_param() is None:  # save model training params
            self.save_param()

        # set default parameters
        if 'dropout_rate' not in para_dict:
            self.para_dict['dropout_rate'] = 0.5
        if 'hidden_layers' not in para_dict:
            self.para_dict['hidden_layers'] = [128, 64, 1]

        # extract hidden layers
        self.n_layers = len(self.para_dict['hidden_layers'])
        self.layer_nodes = self.para_dict['hidden_layers']

    def net_init(self):

        self.fcs = []
        self.fcs.append(nn.Linear(in_features = FP_dim[self.para_dict['feature_type']],
                        out_features = self.layer_nodes[0]))
        for i in range(1, self.n_layers):
            self.fcs.append(nn.Linear(in_features = self.layer_nodes[i-1],
                                      out_features = self.layer_nodes[i]))

    def forward(self, Xs):
        batch_size = len(Xs)

        if self.para_dict['GPU']:
            X = torch.cuda.FloatTensor(Xs)
        else:
            X = torch.FloatTensor(Xs)
        
        X = torch.flatten(X, start_dim = 1)

        for i in range(0, self.n_layers):
            out = F.dropout(F.relu(self.fcs[i]), p = self.para_dict['dropout_rate'])

        return out

    def forward4optim(Xs):
        batch_size = len(Xs)

        if self.para_dict['GPU']:
            X = torch.cuda.FloatTensor(Xs)
        else:
            X = torch.FloatTensor(Xs)
        
        X = torch.flatten(X, start_dim = 1)
        self.X_variable = torch.autograd.Variable(X, requires_grad = True)

        with torch.no_grad():
            for i in range(0, self.n_layers):
                out = F.relu(self.fcs[i]), p = self.para_dict['dropout_rate']

        return out

    def get_gradient(self, Xs, labels):
        out = self.forward4optim(Xs)
        
        loss_func = self.objective()
        optim_x_iter = [self.X_variable].__iter__()
        loss = loss_func(out, torch.FloatTensor(labels))
        loss.backward()

        return self.X_variable.grad
    

    def objective(self):
        return nn.MSELoss()

    def optimizers(self):
        if self.para_dict['optim_name'] == 'Adam':
            return optim.Adam(self.parameters(), lr=self.para_dict['learning_rate'])

        elif self.para_dict['optim_name'] == 'RMSprop':
            return optim.RMSprop(self.parameters(), lr=self.para_dict['learning_rate'])

        elif self.para_dict['optim_name'] == 'SGD':
            return optim.SGD(self.parameters(), lr=self.para_dict['learning_rate'])

    def fit(self, data_loader):
    	self.net_init()
    	saved_epoch = self.load_model()

    	self.train()
    	optimizer = self.optimizers()

    	loss_func = self.objective()
    	for e in range(saved_epoch, self.para_dict['epoch']):
    		total_loss = 0
    		output_train = []

    		for features, labels in data_loader:
    			if self.para_dict['GPU']:
    				y = torch.cuda.FloatTensor(labels)
    		    else:
                    y = torch.FloatTensor(labels)

    			logps = self.forward(features)
    			loss = loss_func(logps, y)
    			total_loss += loss
    			outputs_train.append(logps.detach().numpy())

    			optimizer.zero_grad()
    			loss.backward()
    			optimizer.step()

    	    self.save_model('Epoch_' + str(e+1), self.data_dict())
    	    print('Epoch: %d Loss=%.3f' % (e+1, total_loss))

    	    values = np.concatenate([i for _, i in data_loader])
    	    self.evaluate(np.concatenate(outputs_train), values)
    
    def evaluate(self, outputs, values):
        y_pred = outputs.flatten()
        y_true = values.flatten()
        r2 = r2_score(y_true, y_pred)

        print('Test: R2 score = %.3f' % (r2))

        return r2

    # --------------------------------
    def load_model(self):
        for e in range(self.para_dict['epoch'], 0, -1):
            if os.path.isfile(os.path.join(self.save_path, 'Epoch_' + str(e))):
                # print(os.path.join(self.save_path, 'Epoch_' + str(e)))
                self.load_state_dict(torch.load(os.path.join(self.save_path, 'Epoch_' + str(e))))
                return e
        return 0

    def save_model(self, filename, model):
        torch.save(model, os.path.join(self.save_path, filename))

    # --------------------------------
    def load_param(self, path=None):
        if path == None:
            filepath = os.path.join(self.model_path, 'train_parameters.json')
        else:
            filepath = os.path.join(path, 'train_parameters.json')
        if os.path.exists(filepath):
            return json.load(open(filepath, 'r'))
        return None

    def save_param(self, path=None):
        if path==None:
            filepath = os.path.join(self.model_path, 'train_parameters.json')
        else:
            filepath = os.path.join(path, 'train_parameters.json')
        with open(filepath, 'w') as f:
            json.dump(self.para_dict, f, indent=2)

    # --------------------------------
    def generate_viz_results(self, data_loader, custom_file = None):
    	# maybe need to load a paired viz model

    	# load parameters

    	# prediction for the training set

    	# dim reduction

    	# clustering

    	# derivative/gradient/sensitivity calculation

    	# color mapping

    	# saving results

        return

