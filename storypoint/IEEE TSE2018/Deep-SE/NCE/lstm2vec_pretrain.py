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
from torch.utils.data import Dataset, DataLoader
from NEW_NCE import *


class MyDataset(Dataset):
    def __init__(self, data_x, data_y, labels=None):
        self.data_x = data_x
        self.data_y = data_y
        self.labels = labels
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.labels:
            return self.data_x[idx], self.data_y[idx], self.labels[idx]
        return self.data_x[idx], self.data_y[idx], None



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
    
    training_set = MyDataset(train_x, train_y, labels)
    validation_set = MyDataset(valid_x, valid_y)
    train_generator = DataLoader(training_set, batch_size=50, shuffle=True, num_workers=6)

    model = UnsupEmb(emb_dim=emb_dim, vocab_size=vocab_size, \
        inp_len=inp_len, n_noise=n_noise, Pn=Pn, cuda=CUDA)

    if CUDA:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    for epoch in tqdm(range(NB_EPOCHS)):
        for local_batch_x, local_batch_y, local_labels in train_generator:
            local_batch_x = torch.LongTensor(local_batch_x)
            local_batch_y = torch.LongTensor(local_batch_y)
            local_labels = torch.LongTensor(local_labels)
            if CUDA:
                local_batch_x = local_batch_x.cuda()
                local_batch_y = local_batch_y.cuda()
                local_labels = local_labels.cuda()
            nce_out = model(local_batch_x, local_batch_y)
            loss = NCE_seq_loss(local_labels, nce_out)
            print("loss: {:.4f}".format(loss.data))
            loss.backward()
            optimizer.step()

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
