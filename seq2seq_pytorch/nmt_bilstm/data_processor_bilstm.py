
import pandas as pd
import unicodedata
import re
import numpy as np
import json
from collections import OrderedDict
from torch.utils.data import Dataset
import configure_bilstm as conf
from gensim.models import KeyedVectors
filename = '../Word2VecLibrary/wikipedia-pubmed-and-PMC-w2v.bin'
wrd2vec_model = KeyedVectors.load_word2vec_format(filename, binary=True)
class LanguageIndex():
    def __init__(self, lang):
        """ lang are the list of phrases from each language"""
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set(lang)

        self.create_index()

    def create_index(self):
        #         for phrase in self.lang:
        #             # update with individual tokens
        #             self.vocab.update(phrase.split(' '))

        # sort the vocab
        self.vocab = sorted(self.vocab)

        # add a padding token with index 0
        self.word2idx['<pad>'] = 0
        self.word2idx['<start>'] = 1
        self.word2idx['<end>'] = 2

        # word to index mapping
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 3  # +3 because of pad, start, end token

        # index to word mapping
        for word, index in self.word2idx.items():
            self.idx2word[index] = word

def unicode_to_ascii(s):
    """
    Normalizes latin chars with accent to their canonical decomposition
    """
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    # w = unicode_to_ascii(w.lower().strip())
    w = unicode_to_ascii(w.strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w

def max_length(tensor):
    return max(len(t) for t in tensor)

def pad_sequences(x, max_len):
    padded = np.zeros((max_len), dtype=np.int64)
    if len(x) > max_len: padded[:] = x[:max_len]
    else: padded[:len(x)] = x
    return padded

def create_tensor(src_file, tgt_file,inp_lang,targ_lang):
    # Bladder Cancer Data
    f_src = open(src_file, encoding='UTF-8').read().strip().split('\n')
    f_tgt = open(tgt_file, encoding='UTF-8').read().strip().split('\n')
    dic_data = {'src':f_src,'trgt':f_tgt}
    data = pd.DataFrame(dic_data)
    # Now we do the preprocessing using pandas and lambdas
    data["src"] = data.src.apply(lambda w: '<start> ' + w + ' <end>')
    data["trgt"] = data.trgt.apply(lambda w: preprocess_sentence(w))
    # Vectorize the input and target languages
    input_tensor = [[inp_lang.word2idx[s] for s in src.split(' ')]  for src in data["src"].values.tolist()]
    target_tensor = [[targ_lang.word2idx[s] for s in trgt.split(' ')]  for trgt in data["trgt"].values.tolist()]
    # calculate the max_length of input and output tensor
    max_length_inp, max_length_tar = max_length(input_tensor), max_length(target_tensor)
    # inplace padding
    input_tensor = [pad_sequences(x, max_length_inp) for x in input_tensor]
    target_tensor = [pad_sequences(x, max_length_tar) for x in target_tensor]
    return input_tensor, target_tensor

class MyData(Dataset):
    def __init__(self, X, y, posts):
        self.data = X
        self.target = y
        self.posts = posts
        # TODO: convert this into torch code is possible
        self.length = [np.sum(1 - np.equal(x, 0)) for x in X]

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        posts = self.posts[index]
        x_len = self.length[index]
        return x, y, x_len, posts

    def __len__(self):
        return len(self.data)

def sort_batch(X, y, lengths, posts):
    lengths, indx = lengths.sort(dim=0, descending=True)
    X = X[indx]
    y = y[indx]
    posts = posts[indx]
    return X, y, lengths, posts
    # return X.transpose(0,1), y, lengths, posts # transpose (batch x seq) to (seq x batch)


def gold_seq_extraction(ys,targ_lang):
    target_seq={}
    for usr in range(0,ys.shape[0]): # batch size
        if usr not in target_seq.keys():
            target_seq[usr]=[]
        for j in range(1,ys.shape[1]):
            seq=targ_lang.idx2word[int(ys[usr,j])]
            if seq not in ['<end>', '<pad>','<start>']:
                target_seq[usr].append(seq)
    return [" ".join(x) for x in list(target_seq.values())]

def pred_seq_extraction(outputs,targ_lang):
    pred_seq={}
    for usr in range(0,outputs.shape[1]): # batch size
        if usr not in pred_seq.keys():
            pred_seq[usr]=[]
        for j in range(0,outputs.shape[0]): # max seq length
            seq=targ_lang.idx2word[outputs[j,usr,:].topk(1)[-1][0].item()] # 3rd dimension for approximation, -1 giving the index of maximun probability
            if seq not in ['<end>', '<pad>','<start>']:
                pred_seq[usr].append(seq)
    return [" ".join(x) for x in list(pred_seq.values())]

def read_post_old(input_path):
    g_posts = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            jo = json.loads(line, object_pairs_hook=OrderedDict)
            g_posts.append(jo['posts'])
        batch_month_vect = {}  # [month][user]
        for single_user_post in g_posts:
            for index, row in single_user_post.items():  # 96 months
                #             source_text = str(row)
                #             post_vect = create_post_vector(source_text)  # word_size*200
                if str(index) not in batch_month_vect:
                    batch_month_vect[str(index)] = []
                    batch_month_vect[str(index)].append(str(row).split(" "))
                    # batch_month_vect[str(index)].append(' '.join(str(row).split(' ')[0:10]))  # for bert model. Taking 10 words only
                else:
                    batch_month_vect[str(index)].append(str(row).split(" "))
                    # batch_month_vect[str(index)].append(
                    #     ' '.join(str(row).split(' ')[0:10]))  # for bert model. Taking 10 words only
    return np.dstack(batch_month_vect.values())[0]


def read_post2(input_path):
    g_posts = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            jo = json.loads(line, object_pairs_hook=OrderedDict)
            g_posts.append(jo['posts'])
        batch_month_vect = {}  # [month][user]
        for single_user_post in g_posts:
            for index, row in single_user_post.items():  # 96 months
                #             source_text = str(row)
                #             post_vect = create_post_vector(source_text)  # word_size*200
                if str(index) not in batch_month_vect:
                    batch_month_vect[str(index)] = []
                    mnth_wrd_th = 10
                    month_wrds = str(row).split(" ")[0:mnth_wrd_th]
                    month_wrds += [''] * (mnth_wrd_th - len(month_wrds))
                    mnth_vect = []
                    for wrd in month_wrds:
                        try:
                            word_vector = (wrd2vec_model[wrd]).tolist()
                            mnth_vect.append(word_vector)
                        except KeyError:
                            word_vector = []
                            word_vector += [0] * 200
                            mnth_vect.append(word_vector)
                            continue
                    batch_month_vect[str(index)].append(mnth_vect)

                else:
                    mnth_wrd_th = 10
                    month_wrds = str(row).split(" ")[0:mnth_wrd_th]
                    month_wrds += [''] * (mnth_wrd_th - len(month_wrds))
                    mnth_vect = []
                    for wrd in month_wrds:
                        try:
                            word_vector = (wrd2vec_model[wrd]).tolist()
                            mnth_vect.append(word_vector)
                        except KeyError:
                            word_vector = []
                            word_vector += [0] * 200
                            mnth_vect.append(word_vector)
                            continue
                    batch_month_vect[str(index)].append(mnth_vect)
    return np.array(list(batch_month_vect.values()))
def read_post(input_path):
    g_posts = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            jo = json.loads(line, object_pairs_hook=OrderedDict)
            g_posts.append(jo['posts'])
        batch_month_vect = []  # [user][month]
        for single_user_post in g_posts:
            sngleUsr=[]
            for index, row in single_user_post.items():
                mnth_wrd_th = conf.mnth_wrd_th
                month_wrds = str(row).split(" ")[0:mnth_wrd_th]
                month_wrds += [''] * (mnth_wrd_th - len(month_wrds))
                mnth_vect = []
                for wrd in month_wrds:
                    try:
                        word_vector = (wrd2vec_model[wrd]).tolist()
                        mnth_vect.append(word_vector)
                    except KeyError:
                        word_vector = []
                        word_vector += [0] * 200
                        mnth_vect.append(word_vector)
                        continue
                sngleUsr.append(mnth_vect)
            batch_month_vect.append(sngleUsr)

    return np.array(batch_month_vect)












