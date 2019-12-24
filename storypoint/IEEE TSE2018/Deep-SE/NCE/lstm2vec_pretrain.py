# from keras.layers import *
# from keras.models import Model
# from keras.constraints import *
# from keras.regularizers import *
import gzip
import numpy
#import cPickle
import sys
import _pickle as cPickle
import numpy as np
import load_data
import torch
import noise_dist
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
# from NCE import *
torch.backends.cudnn.enabled = False


def init_weight(modules, activation):
    """
    Weight initialization
    :param modules: Iterable of modules
    :param activation: Activation function.
    """
    for m in modules:
        if isinstance(m, nn.Linear):
            if activation is None:
                m.weight.data = init.xavier_uniform_(m.weight.data) #, gain=nn.init.calculate_gain(activation.lower()))
            else:
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain(activation.lower()))
            if m.bias is not None:
                m.bias.data = init.constant_(m.bias.data, 0.0)


def NCE_seq_loss(y_true, y_pred):
    bceloss = nn.BCELoss()
    loss = bceloss(y_pred, y_true[:, :, 1:])
    import pdb
    pdb.set_trace()


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

    def forward(self, train_x_batch, train_y_batch, train_batch_label):
        train_x_batch = torch.LongTensor(train_x_batch)
        train_y_batch = torch.LongTensor(train_y_batch)
        train_batch_label = torch.LongTensor(train_batch_label)
        if self.is_cuda:
            train_x_batch = train_x_batch.cuda()
            train_y_batch = train_y_batch.cuda()
            train_batch_label = train_batch_label.cuda()

        train_x_emb = self.embedding(train_x_batch)
        GRU_context = self.lstm(train_x_emb)[0]
        nce_out = self.nce(GRU_context, train_y_batch)


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
        return F.sigmoid(out)
    

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


if __name__ == "__main__":
    arg = load_data.arg_passing(sys.argv)
    CUDA=True
    NB_EPOCHS=20
    BATCH_SIZE=50
    dataset = '../data/' + arg['-data'] + '_pretrain.pkl.gz'
    saving = arg['-saving']
    emb_dim = arg['-dim']
    max_len = arg['-len']
    log = 'log/' + saving + '.txt'

    n_noise = 100
    print('Loading data...')
    train, valid, test = load_data.load(dataset)
    valid = valid[-5000:]
    vocab_size = arg['-vocab']

    print('vocab: ', vocab_size)

    ######################################################
    # prepare_lm load data and prepare input, output and then call the prepare_mask function
    # all word idx is added with 1, 0 is for masking -> vocabulary += 1
    train_x, train_y, train_mask = load_data.prepare_lm(train, vocab_size, max_len)
    valid_x, valid_y, valid_mask = load_data.prepare_lm(valid, vocab_size, max_len)


    print('Data size: Train: %d, valid: %d' % (len(train_x), len(valid_x)))

    vocab_size += 1
    n_samples, inp_len = train_x.shape

    # Compute noise distribution and prepare labels for training data: next words from data + next words from noise
    # distribution of words in corpus
    Pn = noise_dist.calc_dist(train, vocab_size)
    Pn = np.array(Pn)

    labels = numpy.zeros((n_samples, inp_len, n_noise + 2), dtype='int64')
    labels[:, :, 0] = train_mask
    labels[:, :, 1] = 1
    
    model = UnsupEmb(emb_dim=emb_dim, vocab_size=vocab_size, \
        inp_len=inp_len, n_noise=n_noise, Pn=Pn, cuda=CUDA)

    if CUDA:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    print("Fucking!!")

    # NUMPY
    train_x_batch = train_x[:BATCH_SIZE]
    train_y_batch = train_y[:BATCH_SIZE]
    train_batch_label = labels[:BATCH_SIZE]

    lol = model(train_x_batch, train_y_batch, train_batch_label)
    # for epoch in tqdm(range(NB_EPOCHS)):
        # optimizer.zero_grad()

        

    print("DONE!")


    
    

"""
if __name__ == "__main__":
    arg = load_data.arg_passing(sys.argv)
    dataset = '../data/' + arg['-data'] + '_pretrain.pkl.gz'
    saving = arg['-saving']
    emb_dim = arg['-dim']
    max_len = arg['-len']
    log = 'log/' + saving + '.txt'

    n_noise = 100
    print('Loading data...')
    train, valid, test = load_data.load(dataset)
    valid = valid[-5000:]
    vocab_size = arg['-vocab']

    print('vocab: ', vocab_size)

    ######################################################
    # prepare_lm load data and prepare input, output and then call the prepare_mask function
    # all word idx is added with 1, 0 is for masking -> vocabulary += 1
    train_x, train_y, train_mask = load_data.prepare_lm(train, vocab_size, max_len)
    valid_x, valid_y, valid_mask = load_data.prepare_lm(valid, vocab_size, max_len)


    print('Data size: Train: %d, valid: %d' % (len(train_x), len(valid_x)))

    vocab_size += 1
    n_samples, inp_len = train_x.shape

    # Compute noise distribution and prepare labels for training data: next words from data + next words from noise
    # distribution of words in corpus
    Pn = noise_dist.calc_dist(train, vocab_size)
    Pn = np.array(Pn)

    labels = numpy.zeros((n_samples, inp_len, n_noise + 2), dtype='int64')
    labels[:, :, 0] = train_mask
    labels[:, :, 1] = 1
    
    print('Building model...')
    # Build model
    main_inp = Input(shape=(inp_len,), dtype='int64', name='main_inp')
    next_inp = Input(shape=(inp_len,), dtype='int64', name='next_inp')

    # Embed the context words to distributed vectors -> feed to GRU layer to compute the context vector
    emb_vec = Embedding(output_dim=emb_dim, input_dim=vocab_size, input_length=inp_len,
                        mask_zero=True)(main_inp)

    GRU_context = LSTM(input_dim=emb_dim, output_dim=emb_dim,
                    return_sequences=True)(emb_vec)

    #GRU_context = Dropout(0.5)(GRU_context)

    # feed output of GRU layer to NCE layer
    nce_out = NCE_seq(input_dim=emb_dim, input_len=inp_len, vocab_size=vocab_size, n_noise=n_noise, Pn=Pn,
                )([GRU_context, next_inp])
    nce_out_test = NCETest_seq(input_dim=emb_dim, input_len=inp_len, vocab_size=vocab_size)([GRU_context, next_inp])

    # Call a model
    model = Model(input=[main_inp, next_inp], output=[nce_out])
    print(model.summary())

    optimizer = RMSprop(lr=0.02, rho=0.99, epsilon=1e-7) #optimizer = RMSprop(lr=0.01)
    model.compile(optimizer=optimizer, loss=NCE_seq_loss)

    testModel = Model(input=[main_inp, next_inp], output=[nce_out_test])
    testModel.compile(optimizer='rmsprop', loss=NCE_seq_loss_test)

    # save result to the filepath and wait if the result doesn't improve after 3 epochs, the lr will be divided by 2
    fParams = 'bestModels/' + saving + '.hdf5'
    callback = NCETestCallback(data=[valid_x, valid_y, valid_mask], testModel= testModel,
                            fResult=log, fParams=fParams)

    json_string = model.to_json()
    fModel = open('models/' + saving + '.json', 'w')
    fModel.write(json_string)

    print('Training...')
    his = model.fit([train_x, train_y], labels,
            batch_size=50, nb_epoch=20,
            callbacks=[callback])

"""
