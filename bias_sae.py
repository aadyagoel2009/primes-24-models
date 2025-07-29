import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
#from torchvision import datasets, transforms
#from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
from torchinfo import summary
import os
#import certifi

from sentence_transformers import SentenceTransformer
from heapq import nlargest
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup #AdamW
from trainer_SAE import SAETrainer

#os.environ['SSL_CERT_FILE'] = certifi.where()

class DecisionMaker(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)
        self.act_out = nn.Sigmoid()

    def forward(self, emb):
        h = self.linear(emb)
        pred = self.act_out(h)
        return pred

class SparseAutoencoder(nn.Module):

    def __init__(self, in_dims, h_dims, sparsity_lambda, sparsity_target=0.05, xavier_norm_init=True):
        super().__init__()
        self.in_dims = in_dims
        self.h_dims = h_dims
        self.sparsity_lambda = sparsity_lambda
        self.sparsity_target = sparsity_target
        self.xavier_norm_init = xavier_norm_init

        """
        Map the original dimensions to a higher dimensional layer of features.
        Apply relu non-linearity to the linear transformation.
        """
        self.encoder = nn.Sequential(
            nn.Linear(self.in_dims, self.h_dims),
            nn.Sigmoid()
        )

        if self.xavier_norm_init:
            nn.init.xavier_uniform_(self.encoder[0].weight)
            nn.init.constant_(self.encoder[0].bias, 0)

        """
        Map back the features to the original input dimensions.
        Apply relu non-linearity to the linear transformation.
        """
        self.decoder = nn.Sequential(
            nn.Linear(self.h_dims, self.in_dims),
            nn.Tanh()
        )

        if self.xavier_norm_init:
            nn.init.xavier_uniform_(self.decoder[0].weight)
            nn.init.constant_(self.decoder[0].bias, 0)

    """
    We pass the original signal through the encoder. Then we pass
    that transformation to the decoder and return both results.
    """
    def forward(self, x, fids):
        encoded = self.encoder(x)
        if fids is not None:
            # rows = []
            # for i in range(len(encoded)):
            #     row = []
            #     for j in range(len(fids)):
            #         row.append(encoded[i][fids[j]])
            #     rows.append(row)
            # act = torch.tensor(rows, device=device, requires_grad=True)
            # return act, None
            return encoded, None
        decoded = self.decoder(encoded)
        #return x, decoded #just to test if original emb can serve the purpose
        return encoded, decoded

    def neutralize_emb(self, emb, dim, neutral_value):
        encoded = self.encoder(emb)
        encoded[:, dim] = float(neutral_value)
        decoded = self.decoder(encoded)
        return decoded

    """
    This is the sparsity penalty we are going to use KL divergence
        - Encourage each hidden neuron to have an average activation (rho_hat) close to the target sparsity level (rho).

    Explanation:
        1. Compute the mean activation of each hidden neuron across the batch
            - We need the average activation to compare it with the target sparsity level. This tells us how active each neuron is on average.

        2. Retrieve the desired average activation level for the hidden neurons.
            - This is the sparsity level we want each neuron to achieve. 
            - Typically a small value like 0.05, meaning we want neurons to be active only 5% of the time.
        
        3.1. Set epsilon constant to prevent division by zero or taking the logarithm of zero.
        3.2. Use torch.clamp to ensure rho_hat stays within the range [epsilon, 1 - epsilon].
            - This is to avoid numerical issues like infinite or undefined values in subsequent calculations.

        4. Calculate the KL divergence between the target sparsity rho and the actual average activation rho_hat for each neuron.
            - rho * torch.log(rho / rho_hat) -> Measures the divergence when the neuron is active.
            - (1 - rho) * torch.log((1 - rho) / (1 - rho_hat)) -> Measures the divergence when the neuron is inactive.
            - The KL divergence quantifies how different the actual activation distribution is from the desired (target) distribution. 
            - A higher value means the neuron is deviating more from the target sparsity level.

        5. Aggregate the divergence values from all hidden neurons to compute a total penalty.
            - We want a single penalty value to add to the loss function, representing the overall sparsity deviation.

        6. Multiply the total KL divergence by a regularization parameter
            - sparsity_lambda controls the weight of the sparsity penalty in the loss function. 
            - A higher value means sparsity is more heavily enforced, while a lower value lessens its impact.
    """
    def sparsity_penalty(self, encoded):
        rho_hat = torch.mean(encoded, dim=0)
        rho = self.sparsity_target
        epsilon = 1e-8
        rho_hat = torch.clamp(rho_hat, min=epsilon, max=1 - epsilon)
        kl_divergence = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
        sparsity_penalty = torch.sum(kl_divergence)
        return self.sparsity_lambda * sparsity_penalty

    """
    Create a custom loss that combine mean squared error (MSE) loss 
    for reconstruction with the sparsity penalty.
    """
    def loss_function(self, x_hat, x, encoded):
        mse_loss = F.mse_loss(x_hat, x)
        sparsity_loss = self.sparsity_penalty(encoded)
        return mse_loss + sparsity_loss

    def plot_distributions(self, x, y):
        plt.hist(x, label='male comments', alpha=.5)
        plt.hist(y, label='female comments', alpha=.5)
        plt.show()

def plot_activations(activations, num_neurons=50, neurons_per_row=10, save_path=None):
    num_rows = (num_neurons + neurons_per_row - 1) // neurons_per_row  
    fig, axes = plt.subplots(num_rows, neurons_per_row, figsize=(neurons_per_row * 2, num_rows * 2))
    axes = axes.flatten()

    for i in range(num_neurons):
        if i >= activations.shape[1]:
            break
        ax = axes[i]
        ax.imshow(activations[:, i].reshape(-1, 1), aspect='auto', cmap='hot')
        ax.set_title(f'Neuron {i+1}', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=6)

    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=600)

    plt.show()

def ask_decision_maker(decision_maker, dm_words, emb_n, enc_model, strengths):
    decision_maker.eval()
    wemb = enc_model.encode(dm_words) if emb_n is None else emb_n.cpu()
    xx = []
    for i in range(len(wemb)):
        input = np.concatenate((wemb[i], np.array([strengths[i]])))
        xx.append(np.array(input))
    xx = torch.tensor(xx, dtype=torch.float32).to(device)
    yy_pred = decision_maker(xx)
    for i in range(len(dm_words)):
        strength = 'good' if strengths[i] > 0 else 'bad'
        score = yy_pred[i].item()
        result = 'qualified' if score >= 0.5 else 'unqualified'
        print(f'For {dm_words[i]}, strength is {strength}, result is {result} with a score {score}.')

def get_data_for_decision_maker(enc_model, test_size):
    x, y, words = [], [], []
    with open('datasets/gendered/word_pairs.txt', 'r') as wpfile:
        for line in wpfile:
            wordm, wordf = line.strip().split(',')
            words.append(wordm)
            words.append(wordf)
    wemb = enc_model.encode(words)
    for i in range(0, len(wemb), 2):
        input = np.concatenate((wemb[i], np.array([5])))
        x.append(np.array(input))
        y.append(np.array([1]))
        input = np.concatenate((wemb[i+1], np.array([-5])))
        x.append(np.array(input))
        y.append(np.array([0]))
    train_idx = int((1-test_size)*len(x))
    x_train, y_train, x_test, y_test = x[:train_idx], y[:train_idx], x[train_idx:], y[train_idx:]
    x_train, x_test = torch.tensor(x_train, dtype=torch.float32).to(device), torch.tensor(x_test, dtype=torch.float32).to(device)
    y_train, y_test = torch.tensor(y_train, dtype=torch.float32).to(device), torch.tensor(y_test, dtype=torch.float32).to(device)
    return x_train, y_train, x_test, y_test

def get_data(enc_model, test_size):
    words = []
    with open('datasets/gendered/word_pairs.txt', 'r') as wpfile:
        for line in wpfile:
            wordm, wordf = line.strip().split(',')
            words.append(wordm)
            words.append(wordf)
    all_data = enc_model.encode(words)
    split_point = int(len(all_data) * (1 - test_size))
    return torch.tensor(all_data[:split_point]), torch.tensor(all_data[split_point:])

def top_neurons(activations, k_top):
    avg_m, avg_f, feature_diff = [], [], dict()
    for j in range(len(activations[0])): #for each neuron (hidden dim)
        sum_m, sum_f = 0, 0
        for i in range(0, len(activations), 2): #for each (m, f) pair
            sum_m += activations[i][j]
            sum_f += activations[i+1][j]
        am, af = sum_m/(len(activations)/2), sum_f/(len(activations)/2) #average over all word-pairs
        avg_m.append(am)
        avg_f.append(af)
        fdiff = am - af if am > af else af - am
        feature_diff[j] = fdiff
    fids = nlargest(k_top, feature_diff, key=feature_diff.get)
    print("Top {} features:".format(k_top))
    for idx in fids:
        print('-- Feature {}, male avg {}, female avg {}'.format(idx, avg_m[idx], avg_f[idx]))
    return fids, avg_m, avg_f

def plot_activation_dists(fids, activations, test_words, acts_test):
    for fid in fids:
        act_m, act_f = [], []
        for i in range(0, len(activations), 2): #for each (m, f) pair
            act_m.append(activations[i][fid])
            act_f.append(activations[i+1][fid])
        act_m, act_f = np.array(act_m), np.array(act_f)
        plt.hist(act_m, label='male words', alpha=.5)
        plt.hist(act_f, label='female words', alpha=.5)
        for i in range(len(test_words)):
            cc = 'k' if i % 7 == 0 else 'g' if i % 7 == 1 else 'r' if i % 7 == 2 else 'c' if i % 7 == 3 else 'm' if i % 7 == 4 else 'y' if i % 7 == 5 else 'b'
            ls = '-' if i % 4 == 0 else '--' if i % 4 == 1 else '-.' if i % 4 == 2 else ':'
            plt.axvline(acts_test[i][fid].cpu(), color=cc, linestyle=ls, linewidth=(i+1)*0.5, label=test_words[i])
        plt.title("Feature "+str(fid))
        plt.xlabel("Activation Value")
        plt.ylabel("Frequency")
        plt.legend(loc='upper left')
        plt.show()

def get_test_set_avg(acts, fids):
    for idx in fids:
        sum_m, sum_f = 0, 0
        for i in range(0, len(acts), 2): #for each (m, f) pair
            sum_m += acts[i][idx]
            sum_f += acts[i+1][idx]
        print('* Test set feature {}, male avg {}, female avg {}'.format(idx, sum_m/(len(acts)/2), sum_f/(len(acts)/2)))

def run_test(test_words, enc_model, sae_model, fids):
    emb = enc_model.encode(test_words)
    data_input = torch.tensor(emb)
    with torch.no_grad():
        acts, _ = sae_model(data_input.to(device), None)
    for i in range(len(test_words)):
        for j in fids:
            print(f'{test_words[i]}: feature {j} value {float(acts[i][j]):.3f}')
    return acts

def get_neutral_act(length, fids, avg_m, avg_f):
    nact = torch.zeros(length, len(avg_m)) #len(fids)
    for i in range(len(fids)):
        neutral_val = (avg_m[fids[i]] + avg_f[fids[i]]) / 2.0
        nact[:, fids[i]] = float(neutral_val)
    return nact

def get_forward_emb(enc_model, debias_words):
    inputs = enc_model.tokenize(debias_words)
    input_ids = inputs['input_ids'].to(device)
    input_attention_mask = inputs['attention_mask'].to(device)
    inputs_final = {'input_ids': input_ids, 'attention_mask': input_attention_mask}
    features = enc_model(inputs_final)
    x = features['sentence_embedding']
    return x

def debias_llm(debias_words, enc_model, sae_model, fids, act_target):
    if not finetune_llm: #just get new emb from decoder
        emb_o = get_forward_emb(enc_model, debias_words) #original embeddings
        with torch.no_grad():
            return sae_model.neutralize_emb(emb_o, fids[0], act_target[0][fids[0]])
    enc_model.to(device)
    enc_model.train()
    sae_model.train()
    # Freeze encoder
    # for name, param in sae_model.named_parameters():
    #     if "encoder" in name:
    #         param.requires_grad = False
    # Define optimizer for trainable parameters
    #sae_trainable_params = [param for param in sae_model.parameters() if param.requires_grad]
    #optimizer_enc = torch.optim.Adam(list(enc_model.parameters()) + sae_trainable_params, lr=0.01)

    #optimizer_enc = torch.optim.Adam(enc_model.parameters(), lr=0.01)
    optimizer_enc = AdamW(enc_model.parameters(), lr=2e-5) #5e-5 0.01
    #optimizer_enc = AdamW(list(enc_model.parameters())+list(sae_model.parameters()), lr=0.01) #5e-5 0.01

    batch_size, d_epochs = 32, 50
    # num_training_steps = d_epochs * len(debias_words)
    # lr_scheduler = get_scheduler(
    #     name="linear", optimizer=optimizer_enc, num_warmup_steps=0, num_training_steps=num_training_steps
    # )
    total_steps = d_epochs * len(debias_words)
    scheduler = get_linear_schedule_with_warmup(optimizer_enc, num_warmup_steps=0, num_training_steps=total_steps)

    print('Train to debias LLM ...')
    for epoch in range(d_epochs):
        total_loss = 0
        #x = enc_model.encode(debias_words)
        x = get_forward_emb(enc_model, debias_words)
        data_input = x.clone().to(device) #using torch.tensor(x) would break the computation graph
        for i in range(0, len(data_input), batch_size): #for data, _ in dataloader: #for each batch
            dbatch = data_input[i:i+batch_size]
            optimizer_enc.zero_grad()
            act, _ = sae_model(dbatch, fids)
            num = batch_size if i+batch_size <= len(data_input) else len(data_input)-i
            #loss = F.mse_loss(act[:num], act_target[i:i+num])
            loss = nn.MSELoss(reduction='none')
            output = loss(act, act_target[i:i+num])
            out1 = torch.sum(output[:, fids[0]])
            out1.backward() #retain_graph=True
            # Implement gradient cliping to prevent explosion
            #torch.nn.utils.clip_grad_norm_(enc_model.parameters(), max_norm=1.0)
            optimizer_enc.step()
            scheduler.step()
            total_loss += out1.item()
        print(f'Epoch: {epoch+1}/{d_epochs} - Train L: {float(total_loss/len(data_input)):.8f}')
        print('-'*64)
    return None

def train_decision_maker(decision_maker, x_train, y_train):
    loss_fn   = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(decision_maker.parameters(), lr=0.01)
    n_epochs = 100 #100 200 300
    batch_size = 300
    decision_maker.train()
    for tep in range(n_epochs):
        eloss = 0
        for i in range(0, len(x_train), batch_size):
            xbatch = x_train[i:i+batch_size]
            y_pred = decision_maker(xbatch)
            ybatch = y_train[i:i+batch_size]
            loss = loss_fn(y_pred, ybatch)
            eloss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #losses.append(eloss*batch_size/len(x_train))
        print(f'-- training epoch {tep+1} of {n_epochs}, loss is {eloss}')
    #print('Training losses are {}'.format(losses))

def topk_diff_dims(emb_set, kdim):
    ndim, np, dim_diff = len(emb_set[0]), len(emb_set)//2, dict()
    adiff = torch.zeros(ndim)
    for i in range(np):
        adiff += emb_set[2*i] - emb_set[2*i+1]
    adiff /= np
    for i in range(ndim):
        dim_diff[i] = abs(adiff[i].item()) #get absolute values
    dims = nlargest(kdim, dim_diff, key=dim_diff.get)
    diffs = [adiff[i].item() for i in dims]
    return dims, diffs

def check_gender_dims(train_set, test_set):
    kdim = 200 #10
    dims, diffs = topk_diff_dims(train_set, kdim)
    print(f'train_set top-{kdim} dimensions are {dims} with differences {diffs}.')
    dims, diffs = topk_diff_dims(test_set, kdim)
    print(f'test_set top-{kdim} dimensions are {dims} with differences {diffs}.')
    #exit()

if __name__ == "__main__":

    def seeding(seed):
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    seeding(1337)

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_epochs', type=int, default=30) #20
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--in_dims', type=int, default=768) #784
    parser.add_argument('--h_dims', type=int, default=256) #5488 2048 256
    parser.add_argument('--sparsity_lambda', type=float, default=1e-6) #1e-4 1e-5 1e-6
    parser.add_argument('--sparsity_target', type=float, default=0.05)
    parser.add_argument('--xavier_norm_init', type=bool, default=True)
    parser.add_argument('--show_summary', type=bool, default=False) #True
    parser.add_argument('--download_mnist', type=bool, default=False) #True
    parser.add_argument('--train', type=bool, default=True) #False
    parser.add_argument('--visualize_activations', type=bool, default=False) #False
    parser.add_argument('--save_model', type=bool, default=False)
    parser.add_argument('--save_plot', type=bool, default=False)
    args = parser.parse_args()
    test_size = 0.5 #0

    #read dataset; get embeddings; ensemble input batch, train/test; train sae; get dim avg; top-k; forward for new input
    print('.. before loading sbert...')
    enc_model = SentenceTransformer("multi-qa-mpnet-base-cos-v1") #all-MiniLM-L6-v2 multi-qa-mpnet-base-cos-v1 paraphrase-MiniLM-L6-v2
    print('.. after loading sbert...')
    train_set, test_set = get_data(enc_model, test_size)
    emb_dims = len(train_set[0])

    check_gender_dims(train_set, test_set)

    sae_model = SparseAutoencoder(
        in_dims=emb_dims, 
        h_dims=args.h_dims, 
        sparsity_lambda=args.sparsity_lambda, 
        sparsity_target=args.sparsity_target,
        xavier_norm_init=args.xavier_norm_init
    )

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print('-' * 64)
    print(f'Using [{str(device).upper()}] for training.\nTo change the device manually, use the argument in the command line.')
    print('-' * 64 + '\n')

    if args.show_summary:
        print('MODEL SUMMARY:')
        summary(sae_model, (emb_dims, True))

    if args.train:
        print('\nTraining...\n')
        trainer = SAETrainer(
            model=sae_model,
            train_set=train_set,
            n_epochs=args.n_epochs,
            lr=args.lr,
            device=device
        )
        activations = trainer.train()
        print('-' * 64)
        print('Trained!')

        if args.visualize_activations:
            print(f'There are {len(activations[0])} neurons in the hidden layer.')
            plot_activations(activations, num_neurons=40, neurons_per_row=10, save_path=None)

            if args.save_plot:
                plot_save_dir = './files'
                os.makedirs(plot_save_dir, exist_ok=True)
                plot_save_path = os.path.join(plot_save_dir, 'activations.png')
                plot_activations(activations, num_neurons=40, neurons_per_row=10, save_path=plot_save_path)

    if args.save_model:
        model_save_dir = './files'
        os.makedirs(model_save_dir, exist_ok=True)
        model_save_path = os.path.join(model_save_dir, 'sae_model.pth')
        torch.save(sae_model.state_dict(), model_save_path)
        print(f'Model saved to {model_save_path}.')
    
    sae_model.eval()
    train_sae_emb, _ = sae_model(train_set.to(device), None)
    test_sae_emb, _ = sae_model(test_set.to(device), None)
    check_gender_dims(train_sae_emb.cpu(), test_sae_emb.cpu())
    #exit()

    #from activations, get dim avg; top-k; test_set result, new input
    k_top = 1
    fids, avg_m, avg_f = top_neurons(activations, k_top)
    finetune_llm = False #False

    if test_size > 0:
        with torch.no_grad():
            acts, _ = sae_model(test_set.to(device), None)
        get_test_set_avg(acts, fids)

    test_words = ['grandma', 'grandpa', 'actress', 'plumber', 'secretary',  'manager', 'professor', 'nurse', 'doctor',
                  'babysitter', 'teacher', 'stay-at-home parent', 'engineer', 'champion']
    #test_words = ['manager', 'nurse', 'stay-at-home parent', 'doctor', 'engineer']
    acts_test = run_test(test_words, enc_model, sae_model, fids)
    plot_activation_dists(fids, activations, test_words, acts_test)

    #train a decision-making predictor with context input and strength input -5/5
    #try engineer, manager, nurse, stay-at-home parent with certain strengths
    test_size = 0.15
    x_train, y_train, x_test, y_test = get_data_for_decision_maker(enc_model, test_size)
    decision_maker = DecisionMaker(len(x_train[0])).to(device)
    print('Training decison maker ...')
    train_decision_maker(decision_maker, x_train, y_train)
    print("Now testing decison maker ...")
    decision_maker.eval()
    y_pred = decision_maker(x_test)
    accuracy = (y_pred.round() == y_test).float().mean() #y_pred.round()
    print(f'Accuracy of decison maker is {accuracy}.')

    dm_words = ['manager', 'nurse', 'doctor', 'stay-at-home parent', 'engineer']
    strengths = [-2, 4, -2, 0.4, -2]
    ask_decision_maker(decision_maker, dm_words, None, enc_model, strengths)

    debias_words = ['manager', 'nurse', 'doctor', 'stay-at-home parent', 'engineer']
    act_target = get_neutral_act(len(debias_words), fids, avg_m, avg_f)
    emb_n = debias_llm(debias_words, enc_model, sae_model, fids, act_target.to(device))

    test_words = ['grandma', 'grandpa', 'actress', 'plumber', 'secretary',  'manager', 'professor', 'nurse', 'doctor',
                  'babysitter', 'teacher', 'stay-at-home parent', 'engineer', 'champion']
    #test_words = ['manager', 'nurse', 'stay-at-home parent', 'doctor', 'engineer']
    acts_test = run_test(test_words, enc_model, sae_model, fids)
    plot_activation_dists(fids, activations, test_words, acts_test)

    #try again dmwords
    ask_decision_maker(decision_maker, dm_words, emb_n, enc_model, strengths)

# python3 bias_sae.py --batch_size 64 --n_epochs 20 --lr 0.0001 --in_dims 784 --h_dims 1024 --sparsity_lambda 1e-5 --sparsity_target 0.05 --xavier_norm_init True --show_summary True --download_mnist True --train False --visualize_activations False --save_model False --save_plot False