# Last Updated: 03/31/2021
post_encoder = True # True / False
post_encoder_type = 'bert' # 'bilstm', 'bert'
g2s = False # True / False
attention = True # True / False
hierarchical_attention = True # True / False
data_forum = 'bladder_cancer' # 'breast_cancer', 'bladder_cancer'
data_type = 'dynamic' # 'static', 'dynamic'(user-month) 'dyn_user'

# Data Specific Configuration
if data_type =='dynamic':
    um_specific = True
else:
    um_specific = False

train_data_path = "../data/"+data_forum+"/train_"+data_type+".data"
dev_data_path = "../data/"+data_forum+"/dev_"+data_type+".data"
test_data_path = "../data/"+data_forum+"/test_"+data_type+".data"
word_idx_file_path = "../data/"+data_forum+"/word.idx"

# Encoder Specific Configuration
if post_encoder_type=='bilstm':
    encoder_axis_2 = 128 #128
elif post_encoder_type == 'bert':
    encoder_axis_2 = 768
post_encoder_batch_dim = 128 #128
post_encoder_keep_prob = 0.5

# Forum Specific Configuration
if data_forum == 'bladder_cancer':
    train_batch_size = 5
    timestamp = 78
    wrd_th = 100
    sample_size_per_layer = 3
    sample_layer_size = 10
    word_embedding_dim = 20
elif data_forum == 'breast_cancer':
    train_batch_size = 50 # users: (train: 2017 dev: 288, test: 577)
    timestamp = 96
    wrd_th=10
    sample_size_per_layer = 10
    sample_layer_size = 4
    word_embedding_dim = 10

# General Configuration
epochs = 50
dev_batch_size = 100
test_batch_size = 100
l2_lambda = 0.0001
learning_rate = 0.001
dropout = 0.0

num_layers_decode = 1 # not used
feature_max_len = 28 # not used

path_embed_method = "lstm" # cnn or lstm or bi-lstm
node_feature_format = "bow" # "bow" or "word_emb"

unknown_word = "<unk>"
PAD = "<PAD>"
GO = "<GO>"
EOS = "<EOS>"
deal_unknown_words = True

seq_max_len = 15

decoder_type = "greedy" # greedy, beam
beam_width = 0
attention_method = "bahdanau" #"bahdanau" (default), "luong", "bahdanau" or "mono_bahdanau" or "mono_luong"

# the following are for the dynamic graph encoding method
weight_decay = 0.00001
hidden_layer_dim = 32 # 32 try 32>
feature_encode_type = "uni"
# graph_encode_method = "max-pooling" # "lstm" or "max-pooling"
graph_encode_direction = "bi" # "single" or "bi"
concat = True

dyn_num_layers = 1
dyn_graph_emb_dim = 128

# multi-task regularization
beta =  0.0003 # 0.0003 # for static=0, regularization
window_size = 6
stride = 1 # 1

encoder = "gated_gcn" # "gated_gcn" "gcn"  "seq"
lstm_in_gcn = "none" # before, after, none