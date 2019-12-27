import gzip
import numpy
import sys
import _pickle as cPickle
import numpy as np
import load_data
import torch
import noise_dist
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
torch.backends.cudnn.enabled = False




def NCE_seq_loss(y_true, y_pred):
    y_true = y_true.float()
    bceloss = nn.BCELoss()
    loss = bceloss(y_pred, y_true[:, :, 1:])
    lol = y_true[:, :, 0] * loss
    return torch.sum(lol) / torch.sum(y_true[:, :, 0])


class UnsupEmb(nn.Module):
    def __init__(self, emb_dim, vocab_size, inp_len, n_noise, Pn, cuda=True):
        super(UnsupEmb, self).__init__()
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.inp_len = inp_len
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.n_noise = n_noise
        self.Pn = Pn
        self.is_cuda = cuda
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=emb_dim, batch_first=True)
        self.nce = NCE_seq(input_dim=emb_dim, input_len=inp_len, vocab_size=vocab_size, n_noise=n_noise, Pn=Pn, cuda=cuda)
        self.nce_test = NCETest_seq(input_dim=emb_dim, input_len=inp_len, vocab_size=vocab_size, cuda=cuda)

    def forward(self, train_x_batch, train_y_batch):
        train_x_emb = self.embedding(train_x_batch)
        GRU_context = self.lstm(train_x_emb)[0]
        nce_out = self.nce(GRU_context, train_y_batch)
        return nce_out

    def test_forward(test_x_batch, test_y_batch, test_batch_label):
        pass



class NCE(nn.Module):
    """
    Noise Contrastive Estimation    
    """
    def __init__(self, init='glorot_uniform', activation='linear',
                 input_dim=None, vocab_size=None, n_noise=25, Pn=np.array([0.5, 0.5]), bias=True, cuda=True, **kwargs):
        super(NCE, self).__init__()   
        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.n_noise = n_noise
        self.Pn = Pn
        self.bias = bias
        self.is_cuda = cuda

        # self.W = nn.Linear(self.vocab_size, self.input_dim, bias=True)
        self.W = nn.Parameter(torch.ones((self.vocab_size, self.input_dim)))
        if self.bias:
            self.b = nn.Parameter(torch.zeros((self.vocab_size)))

        nn.init.xavier_uniform_(self.W.data)



    def forward(self, GRU_context, next_input, mask=None):
        context = GRU_context
        next_w = next_input

        n_samples = next_w.shape[0]
        n_next = self.n_noise + 1

        noise_w = np.random.choice(np.arange(len(self.Pn)), size=(n_samples, self.n_noise), p=self.Pn)
        next_w = np.concatenate([next_w, noise_w], axis=-1)

        next_w = torch.LongTensor(next_w)
        if self.is_cuda:
            next_w = next_w.cuda()

        W_ = self.W[next_w.flatten()].flatten().view([n_samples, n_next, self.input_dim])
        b_ = self.b[next_w.flatten()].view([n_samples, n_next])

        s_theta = (context[:, None, :] * W_).sum(axis=-1) + b_
        noiseP = self.Pn[next_w.flatten()].view([n_samples, n_next])
        noise_score = torch.log(self.n_noise * noiseP)

        out = s_theta - noise_score

        return torch.sigmoid(out)


class NCE_seq(NCE):
    def __init__(self, input_len, **kwargs):
        self.input_len = input_len
        super(NCE_seq, self).__init__(**kwargs)

    def forward(self, GRU_context, next_inp):
        """
        input is torch
        """
        context = GRU_context # torch
        next_w = next_inp # torch

        n_samples, n_steps = next_w.shape
        n_next = self.n_noise + 1

        noise_w = np.random.choice(np.arange(len(self.Pn)), size=(n_samples, n_steps, self.n_noise), p=self.Pn)
        noise_w = torch.LongTensor(noise_w)
        if self.is_cuda:
            noise_w = noise_w.cuda()
        
        next_w = next_w.flatten().view([n_samples, n_steps, 1])
        next_w = torch.cat([next_w, noise_w], dim=-1)

        W_ = self.W[next_w.flatten()].flatten().view([n_samples, n_steps, n_next, self.input_dim])
        b_ = self.b[next_w.flatten()].view([n_samples, n_steps, n_next])
        s_theta = (context[:, :, None, :] * W_).sum(dim=-1) + b_
        noiseP = self.Pn[next_w.flatten().tolist()].reshape([n_samples, n_steps, n_next])
        noiseP = torch.FloatTensor(noiseP)
        if self.is_cuda:
            noiseP = noiseP.cuda()
        noise_score = torch.log(self.n_noise * noiseP)

        out = s_theta - noise_score
        return torch.sigmoid(out)
    

class NCETest(NCE):
    def get_output_shape_for(self, input_shape):
        return (None, 1)
    
    def forward(self, GRU_context, next_inp, mask=None):
        context = GRU_context
        next_w = next_inp

        n_samples = next_w.shape[0]
        context = torch.FloatTensor(context)
        if self.is_cuda:
            context = context.cuda()

        out = torch.matmul(context, self.W.t()) + self.b
        out = torch.softmax(out)
        next_w = next_w.flatten()
        return out[torch.LongTensor(np.arange(n_samples)), next_w]


class NCETest_seq(NCETest):
    def __init__(self, input_len=10, **kwargs):
        self.input_len = input_len
        super(NCETest_seq, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        return (None, self.input_len)
    
    def compute_mask(self, input, mask=None):
        return mask[0]
    
    def forward(self, GRU_context, next_inp, mask=None):
        context = GRU_context
        next_w = next_inp
        n_samples, n_steps = next_w.shape
        vocab_size = self.W.shape[0]

        context = torch.Longtensor(context)
        if self.is_cuda:
            context = context.cuda()

        out = torch.matmul(context, self.W.t()) + self.b
        out = torch.softmax(out)
        out = out.flatten().view([n_samples * n_steps, vocab_size])
        next_w = next_w.flatten()

        prob = out[tensor.arange(n_samples * n_steps), next_w]
        prob = prob.view([n_samples, n_steps])
        return prob

