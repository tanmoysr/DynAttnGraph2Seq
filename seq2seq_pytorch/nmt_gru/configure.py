
post_encodder = True # True, False


train_src = '../data/bladder_cancer/train.src'
train_tgt = '../data/bladder_cancer/train.tgt'
dev_src = '../data/bladder_cancer/dev.src'
dev_tgt = '../data/bladder_cancer/dev.tgt'
test_src = '../data/bladder_cancer/test.src'
test_tgt = '../data/bladder_cancer/test.tgt'
vocab_src = '../data/bladder_cancer/vocab.src'
vocab_tgt = '../data/bladder_cancer/vocab.tgt'

train_post = 'data/bladder_cancer/train_dynamic.data'
dev_post = 'data/bladder_cancer/dev_dynamic.data'
test_post = 'data/bladder_cancer/test_dynamic.data'

# saving location
save_model_path = 'saved_model/'

# Hyper-parameter
epoch = 50
batch_size = 5
embedding_dim = 128 #256, 128
units = 1024
dropout = 0.2 #0.5, 0.2
mnth_wrd_th = 10
