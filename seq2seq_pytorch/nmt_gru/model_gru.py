import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

criterion = nn.CrossEntropyLoss()
def loss_function(real, pred, device):
    mask = real.ge(1).to(device).float()
    loss_ = criterion(pred, real) * mask
    return torch.mean(loss_)

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz,dropout, post_encoder=False):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim, self.enc_units)
        self.dropout = nn.Dropout(dropout)
        # Post Encoder:
        self.post_encoder=post_encoder
        if self.post_encoder:
            self.rnn_mnth = nn.LSTM(200, embedding_dim, batch_first=True, bidirectional=True)
            self.rnn_usr = nn.LSTM(embedding_dim * 2, embedding_dim, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(self.enc_units + 2 * self.embedding_dim, self.enc_units)
    def forward(self, x, lens, posts, device):
        x = self.dropout(self.embedding(x))
        x = pack_padded_sequence(x, lens)  # unpad
        self.hidden = self.initialize_hidden_state(device)
        output, self.hidden = self.gru(x,
                                       self.hidden)  # gru returns hidden state of all timesteps as well as hidden state at last timestep
        # initial decoder hidden is final hidden state of the forwards and backwards
        # pad the sequence to the max length in the batch
        output, _ = pad_packed_sequence(output)
        # post encoder
        if self.post_encoder:
            h0_m = torch.zeros(2, posts.size(0), self.embedding_dim)
            c0_m = torch.zeros(2, posts.size(0), self.embedding_dim)
            month_encoder = None
            for i in range(posts.shape[1]):
                out_l_a, _l = self.rnn_mnth(posts[:, i, :, :], (h0_m, c0_m))
                stacked_hidden_mnth = torch.cat((_l[0][0], _l[0][1]), dim=1)
                if i == 0:
                    month_encoder = stacked_hidden_mnth.view(stacked_hidden_mnth.size()[0], 1,
                                                             stacked_hidden_mnth.size()[1])
                else:
                    month_encoder = torch.cat((month_encoder, stacked_hidden_mnth.view(stacked_hidden_mnth.size()[0], 1,
                                                                                       stacked_hidden_mnth.size()[1])),
                                              dim=1)
            out_u, _u = self.rnn_usr(month_encoder, (h0_m, c0_m))
            stacked_usr = torch.cat((_u[0][0], _u[0][1]), dim=1)
            self.hidden = torch.cat((self.hidden, stacked_usr.view(1, stacked_usr.size()[0], stacked_usr.size()[1])),
                                           dim=2)
            self.hidden =self.fc(self.hidden)
        return output, self.hidden

    def initialize_hidden_state(self, device):
        return torch.zeros((1, self.batch_sz, self.enc_units)).to(device)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, dec_units, enc_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.enc_units = enc_units
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim + self.enc_units,
                          self.dec_units,
                          batch_first=True)
        self.fc = nn.Linear(self.enc_units, self.vocab_size)
        # used for attention
        self.W1 = nn.Linear(self.enc_units, self.dec_units)
        self.W2 = nn.Linear(self.enc_units, self.dec_units)
        self.V = nn.Linear(self.enc_units, 1)

    def forward(self, x, hidden, enc_output):
        enc_output = enc_output.permute(1, 0, 2)
        hidden_with_time_axis = hidden.permute(1, 0, 2)
        score = torch.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis))
        attention_weights = torch.softmax(self.V(score), dim=1)
        context_vector = attention_weights * enc_output
        context_vector = torch.sum(context_vector, dim=1)
        x = self.embedding(x)
        x = torch.cat((context_vector.unsqueeze(1), x), -1)
        # passing the concatenated vector to the GRU
        # output: (batch_size, 1, hidden_size)
        output, state = self.gru(x)
        output = output.view(-1, output.size(2))
        x = self.fc(output)
        return x, state, attention_weights

    def initialize_hidden_state(self):
        return torch.zeros((1, self.batch_sz, self.dec_units))

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, targ_lang, batch_size, device, if_postEncode=False):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.targ_lang = targ_lang
        self.batch_size = batch_size
        self.device = device
        self.post_encoder = if_postEncode

    def forward(self, xs, ys, lens, posts ):
        enc_output, enc_hidden = self.encoder(xs.to(self.device), lens.to(self.device), posts.to(self.device), self.device)
        dec_hidden = enc_hidden
        dec_input = torch.tensor([[self.targ_lang.word2idx['<start>']]] * self.batch_size)
        # tensor to store decoder outputs
        trg_vocab_size = len(self.targ_lang.word2idx)
        outputs = torch.zeros(ys.to(self.device).shape[1], self.batch_size, trg_vocab_size).to(self.device)
        loss = 0
        for t in range(1, ys.to(self.device).size(1)):
            predictions, dec_hidden, _ = self.decoder(dec_input.to(self.device),
                                                 dec_hidden.to(self.device),
                                                 enc_output.to(self.device))
            # place predictions in a tensor holding predictions for each token
            outputs[t] = predictions
            loss += loss_function(ys[:, t].to(self.device), predictions.to(self.device), self.device)
            dec_input = ys[:, t].unsqueeze(1)
        return outputs, loss
