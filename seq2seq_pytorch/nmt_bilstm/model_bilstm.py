import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()
def loss_function(real, pred, device):
    """ Only consider non-zero inputs in the loss; mask needed """
    # mask = 1 - np.equal(real, 0) # assign 0 to all above 0 and 1 to all 0s
    # print(mask)
    #     mask = real.ge(1).type(torch.cuda.FloatTensor)
    mask = real.ge(1).to(device).float()

    loss_ = criterion(pred, real) * mask
    return torch.mean(loss_)

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz,dropout, layer, post_encoder=False):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        # self.gru = nn.GRU(self.embedding_dim, self.enc_units)
        self.bilstm = nn.LSTM(self.embedding_dim, self.embedding_dim, layer, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.post_encoder = post_encoder
        if self.post_encoder:
            self.rnn_mnth = nn.LSTM(200, embedding_dim, batch_first=True, bidirectional=True)
            self.rnn_usr = nn.LSTM(embedding_dim * 2, embedding_dim, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(self.embedding_dim *4, self.embedding_dim)
    def forward(self, x, lens, posts, device):
        # print(x.shape)
        # x = torch.unsqueeze(x, 2)
        # x: batch_size, max_length

        # x: batch_size, max_length, embedding_dim
        x = self.dropout(self.embedding(x))

        # x transformed = max_len X batch_size X embedding_dim
        # x = x.permute(1,0,2)
        # x = pack_padded_sequence(x, lens)  # unpad

        self.hidden = self.initialize_hidden_state(device)

        # output: max_length, batch_size, enc_units
        # self.hidden: 1, batch_size, enc_units
        h0_s = torch.zeros(2, x.size(0), self.embedding_dim) # 2 for bi-lstm
        c0_s = torch.zeros(2, x.size(0), self.embedding_dim)
        # print(x.shape) #[5, 253, 128]
        # print(h0_s.shape) #[2, 253, 128]
        output, _hidden = self.bilstm(x, (h0_s, c0_s))
        self.hidden = torch.cat((_hidden[0][0], _hidden[0][1]), dim=1)
        # output, self.hidden = self.lstm(x)
        # output, self.hidden = self.gru(x,
        #                                self.hidden)  # gru returns hidden state of all timesteps as well as hidden state at last timestep

        # initial decoder hidden is final hidden state of the forwards and backwards
        # pad the sequence to the max length in the batch
        # output, _ = pad_packed_sequence(output)

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
            self.hidden = torch.cat((self.hidden, stacked_usr), dim=1)
            # self.hidden = torch.cat((self.hidden, stacked_usr.view(1, stacked_usr.size()[0], stacked_usr.size()[1])),
            #                                dim=2)

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
        # self.gru = nn.GRU(self.embedding_dim + self.enc_units,
        #                   self.dec_units,
        #                   batch_first=True)
        self.lstm = nn.LSTM(self.embedding_dim*3,
                          self.embedding_dim,
                          batch_first=True)
        self.fc = nn.Linear(self.embedding_dim, self.vocab_size)

        # used for attention
        self.W1 = nn.Linear(self.embedding_dim*2, self.dec_units)
        self.W2 = nn.Linear(self.embedding_dim, self.dec_units)
        self.V = nn.Linear(self.enc_units, 1)

    def forward(self, x, hidden, enc_output):
        # enc_output original: (max_length, batch_size, enc_units)
        # enc_output converted == (batch_size, max_length, hidden_size)
        # enc_output = enc_output.permute(1, 0, 2)
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score

        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # hidden_with_time_axis = hidden.permute(1, 0, 2)
        hidden_with_time_axis = hidden.unsqueeze(1)

        # score: (batch_size, max_length, hidden_size)
        # print(enc_output.shape, hidden_with_time_axis.shape) # [batch_size, max_len, enc_unit], [batch_size, 1, enc_unit]
        score = torch.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis))

        # score = torch.tanh(self.W2(hidden_with_time_axis) + self.W1(enc_output))

        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = torch.softmax(self.V(score), dim=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_output
        context_vector = torch.sum(context_vector, dim=1)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        # print(x.shape)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        # x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # ? Looks like attention vector in diagram of source
        x = torch.cat((context_vector.unsqueeze(1), x), -1)
        # print(x.shape)

        # passing the concatenated vector to the GRU
        # output: (batch_size, 1, hidden_size)
        # output, state = self.gru(x)
        output, state = self.lstm(x)
        # print(output.shape)

        # output shape == (batch_size * 1, hidden_size)
        output = output.view(-1, output.size(2))

        # output shape == (batch_size * 1, vocab)
        x = self.fc(output)

        # print(x.shape)
        # print(state[0].shape)
        # state = state.permute(1, 0, 2)

        return x, torch.squeeze(state[0]), attention_weights

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
        # print(enc_output.shape, enc_hidden.shape)
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
