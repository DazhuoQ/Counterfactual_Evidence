import random
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv
from torch.nn import Sequential as Seq, Linear, ReLU, BatchNorm1d
import torch.optim as optim

from src.utils import *

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return F.log_softmax(x, dim=1)
    
class GCN2(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN2, self).__init__()
        self.conv1 = GCNConv(input_dim, input_dim)
        self.conv2 = GCNConv(input_dim, input_dim)
        self.conv3 = GCNConv(input_dim, input_dim)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return x
    

class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        # First GraphSAGE layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        # Second GraphSAGE layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        # Third GraphSAGE layer
        x = self.conv3(x, edge_index)
        
        return F.log_softmax(x, dim=1)

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class MPNN(MessagePassing):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MPNN, self).__init__(aggr='add')  # Aggregation is 'add', can also be 'mean' or 'max'
        self.lin1 = torch.nn.Linear(input_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        # Add self-loops to the adjacency matrix
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Apply the first linear transformation
        x = self.lin1(x)
        
        # Perform message passing
        x = self.propagate(edge_index, x=x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        # Second message passing layer
        x = self.lin2(x)
        x = self.propagate(edge_index, x=x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        # Output layer
        x = self.lin3(x)
        
        return F.log_softmax(x, dim=1)

    def message(self, x_j, edge_index, size):
        # x_j is the feature of the neighboring nodes
        return x_j

    def update(self, aggr_out):
        # Update node embedding after aggregation
        return aggr_out



class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GAT, self).__init__()
        heads = 8

        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=0.6)
        self.conv3 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=True, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=-1)


class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GIN, self).__init__()

        self.conv1 = GINConv(Seq(Linear(input_dim, hidden_dim),
                                 ReLU(),
                                 Linear(hidden_dim, hidden_dim),
                                 ReLU(),
                                 BatchNorm1d(hidden_dim)), train_eps=True)
        
        self.conv2 = GINConv(Seq(Linear(hidden_dim, hidden_dim),
                                 ReLU(),
                                 Linear(hidden_dim, hidden_dim),
                                 ReLU(),
                                 BatchNorm1d(hidden_dim)), train_eps=True)
        
        self.conv3 = GINConv(Seq(Linear(hidden_dim, hidden_dim),
                                 ReLU(),
                                 Linear(hidden_dim, hidden_dim),
                                 ReLU(),
                                 BatchNorm1d(hidden_dim)), train_eps=True)
        
        self.lin = Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        return self.lin(x)


def get_model(config):
    model_name = config['model_name']
    input_dim = config['input_dim']
    hidden_dim = config['hidden_dim']
    output_dim = config['output_dim']
    data_name = config['data_name']

    if model_name == 'gcn':
        model = GCN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        return model
    elif model_name == 'gat':
        model = GAT(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        return model 
    elif model_name == 'gin':
        model = GIN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        return model 
    elif model_name == "sage":
        model = GraphSAGE(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        return model
    elif model_name == "mpnn":
        model = MPNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        return model



def main(config_file, output_dir):
    # Load configuration
    config = load_config(config_file)
    data_name = config['data_name']
    model_name = config['model_name']
    random_seed = config['random_seed']
    set_seed(config['random_seed'])

    # Device config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Get input graph
    data = dataset_func(config, random_seed)
    data.to(device)

    # Get the model for training
    model = get_model(config)
    model.to(device)
    best_loss = float('inf')
    best_model = None

    # train and save the model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    model.train()
    if model_name =="fairgnn":
        for epoch in range(1000):
            model.optimize(data.x, data.edge_index, data.y, data.idx_train, data.sens, data.idx_sens_train)
            print(f'Epoch {epoch+1}, Loss: {model.G_loss.detach().item()}')
            if model.G_loss < best_loss:
                best_loss = model.G_loss.detach().item()
                best_model = model.state_dict()
    else:
        for epoch in range(1000):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            if model_name=='gin':
                loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            else:
                loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_model = model.state_dict()
    
            
    torch.save(best_model, 'models/{}_{}_model.pth'.format(data_name, model_name))

    # Save experiment settings
    print('Seed: '+str(config['random_seed']))
    print('Dataset: '+str(config['data_name']))
    print('Model: '+str(config['model_name']))

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python model.py <config_file> <output_dir>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    output_dir = sys.argv[2]
    main(config_file, output_dir)

