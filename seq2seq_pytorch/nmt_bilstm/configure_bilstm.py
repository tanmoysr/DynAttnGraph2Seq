forum_name = 'bladder_cancer' # 'breast_cancer', 'bladder_cancer'
if forum_name == 'bladder_cancer':
    mnth_wrd_th = 100
elif forum_name == 'breast_cancer':
    mnth_wrd_th = 10

train_src = 'data/'+forum_name+'/train.src'
train_tgt = 'data/'+forum_name+'/train.tgt'
dev_src = 'data/'+forum_name+'/dev.src'
dev_tgt = 'data/'+forum_name+'/dev.tgt'
test_src = 'data/'+forum_name+'/test.src'
test_tgt = 'data/'+forum_name+'/test.tgt'
vocab_src = 'data/'+forum_name+'/vocab.src'
vocab_tgt = 'data/'+forum_name+'/vocab.tgt'

post_encodder = True # True, False
# train_post = 'data/bladder_cancer/dev_dynamic.data'
# dev_post = 'data/bladder_cancer/dev_dynamic.data'
# test_post = 'data/bladder_cancer/dev_dynamic.data'
train_post = 'data/'+forum_name+'/train_dynamic.data'
dev_post = 'data/'+forum_name+'/dev_dynamic.data'
test_post = 'data/'+forum_name+'/test_dynamic.data'
# saving location
save_model_path = 'saved_model/'
# Hyper-parameter
epoch = 1
batch_size = 5
embedding_dim = 128 #256, 128
units = 1024
dropout = 0.2 #0.5, 0.2
learning_rate = 1 #0.001, 1
layer=1

