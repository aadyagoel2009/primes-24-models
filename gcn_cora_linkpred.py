from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T
from torch_geometric.data import Data

neg_edge = True
transform = T.RandomLinkSplit(
    num_val = 0.15,
    num_test = 0.15,
    add_negative_train_samples = True,
    neg_sampling_ratio = 1.0,
    )

num_epoch_lp = 500

# https://graphsandnetworks.com/the-cora-dataset/
dataset = Planetoid(root='/tmp/Cora', name='Cora')
print('Dataset info:')
print(dataset)
num_node_features = dataset.num_node_features
dim_hidden = 16
dim_output = dataset.num_classes
num_epoch = 200
learning_rate = 0.01
print('-----------------------')
print('The number of node features: {}\nThe dimension of hidden layer: {}\nThe dimension of output layer: {}'.format(num_node_features, dim_hidden, dim_output))
print('-----------------------')

class linkpred(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(dim_hidden*2, dim_hidden)
        self.l2 = torch.nn.Linear(dim_hidden, 1)
    def forward(self, x, edge_label_index):
        edge_head = x[edge_label_index[0]]
        edge_tail = x[edge_label_index[1]]
        headtail_emb = torch.cat((edge_head, edge_tail), 1)
        h = self.l1(headtail_emb)
        h = F.relu(h)
        h = self.l2(h)
        pred = torch.sigmoid(h)
        return pred
        
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, dim_hidden)
        self.conv2 = GCNConv(dim_hidden, dim_hidden)
        self.linear = torch.nn.Linear(dim_hidden, dim_output)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index = edge_index.T[1:]
        edge_index = edge_index.T
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        emb = x.clone()
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.linear(x)
        return F.log_softmax(x, dim=1), emb
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
model = GCN().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
loss_log = []

model.train()
for epoch in range(num_epoch):
    optimizer.zero_grad()    
    output, _ = model(data)
    loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
    loss_log.append(loss.item())
    loss.backward()   
    optimizer.step()


model.eval()
output, emb = model(data)
pred = output.argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
accuracy = int(correct) / int(data.test_mask.sum())
print('Number of test sample:{}\nNumber of correct prediction:{}\nThe accuracy:{:.4f}'.format(data.test_mask.sum(),correct,accuracy))

model_lp = linkpred().to(device)
optimizer_lp = torch.optim.Adam(model_lp.parameters(), lr=learning_rate, weight_decay=5e-4)
data_lp = Data(x=data.x, edge_index=data.edge_index)
if neg_edge == True:
    lp_train, lp_val, lp_test = transform(data_lp) 

lp_train = lp_train.to(device)
lp_val = lp_val.to(device)
lp_test = lp_test.to(device)
loss_log_lp = []


model_lp.train()  
for epoch in range(num_epoch_lp):
    optimizer_lp.zero_grad()
    _, x = model(lp_train) 
    x = x.detach() 
    pred = model_lp(x, lp_train.edge_label_index) 
    pred = pred.squeeze(1)
    loss = F.binary_cross_entropy_with_logits(pred, lp_train.edge_label)
    loss_log_lp.append(loss.item())
    loss.backward()
    optimizer_lp.step()


model_lp.eval() 
_, x = model(lp_val) 
x = x.detach()
val_pred = model_lp(x, lp_val.edge_label_index) 
val_pred = val_pred.squeeze(1)
correct = (val_pred.round() == lp_val.edge_label).sum()

accuracy = int(correct) / (len(lp_val.edge_label))

print(f"Link prediction test accuracy: {accuracy}")