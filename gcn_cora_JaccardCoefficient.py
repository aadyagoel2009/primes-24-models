from torch_geometric.datasets import Planetoid
import torch as th
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T 
from torch_geometric.data import Data 
from random import randrange

def th_delete(e_index, indices):
    mask = th.ones(e_index.size(dim=0), dtype=th.bool)
    mask[indices] = False
    return e_index[mask, :]

def neighbors(v, edge_index):
    nei = set()
    for i in range(len(edge_index[0])):
        if edge_index[0][i] == v:
            nei.add(edge_index[1][i].item())
        if edge_index[1][i] == v:
            nei.add(edge_index[0][i].item())
    return nei

def jaccard(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if len(union) == 0:
        return 0
    return len(intersection) / len(union)

neg_edge = True 
transform = T.RandomLinkSplit(
    num_val = 0.15,
    num_test = 0.15,
    add_negative_train_samples = True,
    neg_sampling_ratio = 1.0,
    )

num_epoch_lp = 500

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

class linkpred(th.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = th.nn.Linear(dim_hidden*2, dim_hidden) 
        self.l2 = th.nn.Linear(dim_hidden, 1)
    def forward(self, x, edge_label_index):
        edge_head = x[edge_label_index[0]]
        edge_tail = x[edge_label_index[1]]
        headtail_emb = th.cat((edge_head, edge_tail), 1)
        h = self.l1(headtail_emb)
        h = F.relu(h)
        h = self.l2(h)
        pred = th.sigmoid(h)
        return pred
        
class GCN(th.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, dim_hidden)
        self.conv2 = GCNConv(dim_hidden, dim_hidden)
        self.linear = th.nn.Linear(dim_hidden, dim_output) 
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if not model_editing:
            edge_index = edge_index.T
            edge_index = th_delete(edge_index, unlearned_edges)
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
        
device = th.device('cuda' if th.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = dataset[0].to(device)
optimizer = th.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
loss_log = []
num_unlearned = 3
unlearned_edges = []
for i in range(num_unlearned):
    unlearned_edges.append(randrange(data.train_mask.size(dim=0))) 
unlearned_edges = [2690, 1823, 2079] 
print('Unlearned edges indices are {}'.format(unlearned_edges))
model_editing = True 

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
optimizer_lp = th.optim.Adam(model_lp.parameters(), lr=learning_rate, weight_decay=5e-4)
data_lp = Data(x=data.x, edge_index=data.edge_index) #for negative sampling; take the original node attributes as node features
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
    pred = model_lp(x, lp_train.edge_label_index) #call link prediction model's forward
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

print(f"Edge prediction test accuracy: {accuracy}")

for un_e in unlearned_edges:
    unlearned_head, unlearned_tail = data.edge_index[0][un_e], data.edge_index[1][un_e]
    random_head, random_tail = randrange(data.x.size(dim=0)), randrange(data.x.size(dim=0))
    e_index = th.tensor([[unlearned_head, random_head], [unlearned_tail, random_tail]])
    eprob = model_lp(x, e_index)
    print('Unlearned edge index {} probability is {:.4f}, edge probability of a random pair of nodes is {:.4f}'.format(un_e, eprob[0][0], eprob[1][0]))

if model_editing:
    print('Unlearning based on model editing...')
    model.train()
    tune_epoch, loss1_log, loss2_log = 3, [], []
    heads, tails, c = [], [], []
    c_base = 0.5
    for un_e in unlearned_edges:
        head = data.edge_index[0][un_e]
        tail = data.edge_index[1][un_e]
        heads.append(head)
        tails.append(tail)
        h_nei, t_nei = neighbors(head, data.edge_index), neighbors(tail, data.edge_index)
        h_nei.discard(tail.item())
        t_nei.discard(head.item())
        jac = jaccard(h_nei, t_nei)
        c.append(jac)
    e_index = th.tensor([heads, tails])
    for epoch in range(tune_epoch):
        optimizer.zero_grad()
        output, x = model(data) 
        loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
        loss1_log.append(loss.item())
        pred = model_lp(x, e_index)
        pred = pred.squeeze(1)
        pred2 = pred.clone().detach()
        for i in range(pred2.size(dim=0)):
            # method 1
            #if pred2[i] > 0.1:
            #  pred2[i] = pred2[i] - 0.1

            # method 2
            # pred2[i] = pred2[i] * 0.9

            # method 3
            pred2[i] = pred2[i] * c_base+ c[i]*(1-c_base)
        
        loss2 = F.binary_cross_entropy_with_logits(pred, pred2)
        loss2_log.append(loss2.item())
        loss += loss2
        loss.backward()
        optimizer.step()
        print('Done unlearning epoch {}'.format(epoch))

    model.eval()
    output, emb = model(data)
    pred = output.argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    accuracy = int(correct) / int(data.test_mask.sum())
    print('Number of test sample:{}\nNumber of correct prediction:{}\nThe accuracy:{:.4f}'.format(data.test_mask.sum(),correct,accuracy))

    for un_e in unlearned_edges:
        unlearned_head, unlearned_tail = data.edge_index[0][un_e], data.edge_index[1][un_e]
        random_head, random_tail = randrange(data.x.size(dim=0)), randrange(data.x.size(dim=0))
        e_index = th.tensor([[unlearned_head, random_head], [unlearned_tail, random_tail]])
        eprob = model_lp(x, e_index)
        print('Unlearned edge index {} probability is {:.4f}, edge probability of a random pair of nodes is {:.4f}'.format(un_e, eprob[0][0], eprob[1][0]))
