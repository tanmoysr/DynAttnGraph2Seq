from gensim.models import KeyedVectors
import tensorflow as tf
import tokenization
import numpy as np
import configure as conf

# Load word vector corpus
# filename = '../Word2VecLibrary/GoogleNews-vectors-negative300.bin'
filename = '../Word2VecLibrary/wikipedia-pubmed-and-PMC-w2v.bin'
wrd2vec_model = KeyedVectors.load_word2vec_format(filename, binary=True)
tfidf_words=[]
input_path="../data/bladder_cancer/word.idx"
with open(input_path, 'r') as f:
    lines = f.readlines()
    for line in lines[3:]:
        word = line.strip().split(' ')[0]
        if word not in tfidf_words:
            tfidf_words.append(word)
# import model
def create_tokenizer(vocab_file='vocab.txt', do_lower_case=False):
    return tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

def convert_sentence_to_features(sentence, tokenizer, max_seq_len):
    tokens = ['[CLS]']
    tokens.extend(tokenizer.tokenize(sentence))
    if len(tokens) > max_seq_len - 1:
        tokens = tokens[:max_seq_len - 1]
    tokens.append('[SEP]')

    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # Zero Mask till seq_length
    zero_mask = [0] * (max_seq_len - len(tokens))
    input_ids.extend(zero_mask)
    input_mask.extend(zero_mask)
    segment_ids.extend(zero_mask)

    return input_ids, input_mask, segment_ids

def convert_sentences_to_features(sentences, tokenizer, max_seq_len=20):
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []

    for sentence in sentences:
        input_ids, input_mask, segment_ids = convert_sentence_to_features(sentence, tokenizer, max_seq_len)
        all_input_ids.append(input_ids)
        all_input_mask.append(input_mask)
        all_segment_ids.append(segment_ids)

    return all_input_ids, all_input_mask, all_segment_ids

def bert_encoding(sess,model,batch_month_vect):
    tokenizer = create_tokenizer()
    encoded_batch_prestate_all_month = None
    for month_index in range(conf.timestamp):
        month_index = str(month_index)
        input_ids_vals, input_mask_vals, segment_ids_vals = convert_sentences_to_features(batch_month_vect[month_index], tokenizer, 20)

        encoded_batch_prestate = sess.run(model.bert_outputs,
                       feed_dict={model.input_ids: input_ids_vals, model.input_mask: input_mask_vals, model.segment_ids: segment_ids_vals})['pooled_output']
        if encoded_batch_prestate_all_month is None:
            encoded_batch_prestate_all_month = encoded_batch_prestate
        else:
            # print(encoded_batch_prestate_all_month.shape, encoded_batch_prestate.shape )
            encoded_batch_prestate_all_month = np.dstack((encoded_batch_prestate_all_month, encoded_batch_prestate))
    # sess2.close()
    return encoded_batch_prestate_all_month

#--------------Bilstm Endoer-----------------
def create_batch_month_vect(batch_posts):
    batch_month_vect = {}  # [month][user]
    for single_user_post in batch_posts:
        #         user_post_vect = []

        for index, row in single_user_post.items():  # 96 months
            #             source_text = str(row)
            #             post_vect = create_post_vector(source_text)  # word_size*200
            if str(index) not in batch_month_vect:
                batch_month_vect[str(index)] = []
                batch_month_vect[str(index)].append(str(row))
                # batch_month_vect[str(index)].append(' '.join(str(row).split(' ')[0:10]))  # for bert model. Taking 10 words only
            else:
                batch_month_vect[str(index)].append(str(row))
                # batch_month_vect[str(index)].append(
                # ' '.join(str(row).split(' ')[0:10]))  # for bert model. Taking 10 words only
    return batch_month_vect

def create_batch_month_vect_g2s(batch_posts):
    batch_month_vect = {}  # [month][user]
    batch_month_vect[str(0)] = []
    for single_user_post in batch_posts:
        user_post_vect = []

        for index, row in single_user_post.items():  # 1 month
            user_post_vect.append(row)
        user_post_vect_join = ' '.join(user_post_vect)
        batch_month_vect[str(0)].append(user_post_vect_join)
    return batch_month_vect

def batch_month_vectfn_tfidf(givenList, batch_monthwise_vect):
    singM_allU_vect = []
    for userIndx in givenList:
        singU_month_base = batch_monthwise_vect[userIndx].split(" ")
        singU_month=[]
        #--- only if you want threshold start----------------
        word_threshold = conf.wrd_th # 10, train data: 36989
        wrd_th=0
        for wrd in singU_month_base:
            if wrd in tfidf_words:
                wrd_th+=1
                if wrd_th<word_threshold:
                    singU_month.append(wrd)
                else:
                    break
        # if len(singU_month)>word_threshold: #
        #     singU_month = singU_month[0:word_threshold]
        # else:
        #     singU_month = singU_month
        # --- only if you want threshold end---------------

        singleU_singleM_vect = []
        # for singU_month_words in set(singU_month.split(" ")):
        # for singU_month_words in singU_month.split(" "):
        for singU_month_words in singU_month:
            try:
                word_vector = (wrd2vec_model[
                    singU_month_words]).tolist()  # increase the dimensionality so that unk words can be defined
                singleU_singleM_vect.append(word_vector)
            except KeyError:
                continue

        singM_allU_vect.append(singleU_singleM_vect)  # user
        maxLength = max(len(x) for x in singM_allU_vect)
        month_array = np.array(singM_allU_vect)
        #             print(len(month_array[82]))
        #             print(maxLength)
        if maxLength != 0:
            for index in range(month_array.shape[0]):
                if len(month_array[index]) == 0:
                    month_array[index] = np.zeros([maxLength, 200], dtype=float)
                if len(month_array[index]) < maxLength:
                    month_array[index] = np.concatenate(
                        ([month_array[index], np.zeros([(maxLength - len(month_array[index])), 200], dtype=float)]),
                        axis=0)
    return list(month_array)

def batch_month_vectfn(givenList, batch_monthwise_vect):
    singM_allU_vect = []
    for userIndx in givenList:
        singU_month = batch_monthwise_vect[userIndx].split(" ")

        #--- only if you want threshold start----------------
        word_threshold = 100 # 10, train data: 36989
        if len(singU_month)>word_threshold: #
            singU_month = singU_month[0:word_threshold]
        else:
            singU_month = singU_month
        # --- only if you want threshold end---------------
        singleU_singleM_vect = []
        # for singU_month_words in set(singU_month.split(" ")):
        # for singU_month_words in singU_month.split(" "):
        for singU_month_words in singU_month:

            try:

                word_vector = (wrd2vec_model[
                    singU_month_words]).tolist()  # increase the dimensionality so that unk words can be defined
                singleU_singleM_vect.append(word_vector)
            except KeyError:

                continue

        singM_allU_vect.append(singleU_singleM_vect)  # user
        maxLength = max(len(x) for x in singM_allU_vect)
        month_array = np.array(singM_allU_vect)
        #             print(len(month_array[82]))
        #             print(maxLength)
        if maxLength != 0:
            for index in range(month_array.shape[0]):
                if len(month_array[index]) == 0:
                    month_array[index] = np.zeros([maxLength, 200], dtype=float)
                if len(month_array[index]) < maxLength:
                    month_array[index] = np.concatenate(
                        ([month_array[index], np.zeros([(maxLength - len(month_array[index])), 200], dtype=float)]),
                        axis=0)
    return list(month_array)


def selection_sort(data, givenIndexes):
    for i in range(len(givenIndexes)):
        swap = i + np.argmin(givenIndexes[i:])
        (data[i], data[swap]) = (data[swap], data[i])
    return data


def encoded_batch_prestatefn(sess, batch_month_vect, batch_length, model):
    encoded_batch_prestate_all_month = None
    for month_index in range(conf.timestamp):
        month_index = str(month_index)
        encoded_batch_prestate = None
        indexes = list(range(len(batch_month_vect[month_index])))
        # sort indexes by frequency, higher -> lower
        indexes.sort(key=lambda i: len(batch_month_vect[month_index][i]) / len(batch_month_vect[month_index]),
                     reverse=True)  # to get the freq higher to lower

        for start in range(0, len(batch_month_vect[month_index]), batch_length):
            #     for start in range(0, 9, 5):

            end = min(start + (batch_length - 1), (len(batch_month_vect[month_index]) - 1))
            givenList = indexes[start:(end + 1)]
            singleM_allU_data = batch_month_vectfn(givenList, batch_month_vect[month_index])
            if len(singleM_allU_data[0]) == 0:
                encoded_month_state_cumulative = np.zeros([len(singleM_allU_data), conf.post_encoder_batch_dim], dtype=int)
            else:
                # if len(givenList)==1:
                #     encoded_month_state_cumulative = np.expand_dims(np.zeros([len(singleM_allU_data), 128], dtype=int), axis=0)
                # else:
                # print(singleM_allU_data)
                reshape_axis_11 = len(singleM_allU_data)  # user
                #             print(reshape_axis_11)
                # reshape_axis_12 = len(singleM_allU_data[1])  # word
                reshape_axis_12 = len(singleM_allU_data[0])  # word for batch size=1
                #             print(reshape_axis_12)
                encoded_month_state = (sess.run(model.encoder1_output,
                                                {model.month_vect_inputs: singleM_allU_data, model.reshape_axis11: reshape_axis_11,
                                                 model.reshape_axis12: reshape_axis_12})[1])
                # (2x96x128)
                #             print(encoded_month_state.shape) #2xuserx128
                #             encoded_month_state_cumulative = None
                encoded_month_state_cumulative = np.sum(encoded_month_state, axis=0)  # userx128
            #             print(encoded_month_state_cumulative.shape)

            # #----------------------------------------------------------------------------------------------
            if encoded_batch_prestate is None:
                encoded_batch_prestate = encoded_month_state_cumulative
            else:
                encoded_batch_prestate = np.concatenate(([encoded_batch_prestate, encoded_month_state_cumulative]),
                                                        axis=0)
        encoded_batch_prestate = selection_sort(encoded_batch_prestate, indexes)
        if encoded_batch_prestate_all_month is None:
            encoded_batch_prestate_all_month = encoded_batch_prestate
        else:
            encoded_batch_prestate_all_month = np.dstack((encoded_batch_prestate_all_month, encoded_batch_prestate))
    return encoded_batch_prestate_all_month



def user_encoder(input_data, num_layers, rnn_size, sequence_length, keep_prob, reshape_axis_2):

    output = tf.reshape(input_data, [reshape_axis_2, conf.timestamp, conf.encoder_axis_2])

    #     layer = 1
    for layer in range(3):
        with tf.variable_scope('encoder_2{}'.format(layer), reuse=tf.AUTO_REUSE):
            cell_fw = tf.contrib.rnn.LSTMCell(64, initializer=tf.truncated_normal_initializer(-0.1, 0.1, seed=2))
            #             cell_fw = tf.contrib.rnn.LSTMCell(64, state_is_tuple=True,  initializer=tf.truncated_normal_initializer(-0.1, 0.1, seed=2))
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=keep_prob)

            cell_bw = tf.contrib.rnn.LSTMCell(64, initializer=tf.truncated_normal_initializer(-0.1, 0.1, seed=2))
            #             cell_bw = tf.contrib.rnn.LSTMCell(64, state_is_tuple=True,  initializer=tf.truncated_normal_initializer(-0.1, 0.1, seed=2))
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=keep_prob)

            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, output, sequence_length,
                                                              dtype=tf.float32)
            output = tf.concat(outputs, 2)
            state = tf.concat(states, 2)

    return [output, state]


def month_encoder(input_data1, num_layers, rnn_size, sequence_length, keep_prob, reshape_axis11, reshape_axis12):
    inputs = tf.reshape(input_data1, [reshape_axis11, reshape_axis12, 200])  # userxwordxvector
    #     layer = 1
    for layer in range(4):
        with tf.variable_scope('encoder_{}'.format(layer), reuse=tf.AUTO_REUSE):
            cell_fw = tf.contrib.rnn.LSTMCell(64, initializer=tf.truncated_normal_initializer(-0.1, 0.1, seed=2))
            #             cell_fw = tf.contrib.rnn.LSTMCell(64, state_is_tuple=True,  initializer=tf.truncated_normal_initializer(-0.1, 0.1, seed=2))
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=keep_prob)

            cell_bw = tf.contrib.rnn.LSTMCell(64, initializer=tf.truncated_normal_initializer(-0.1, 0.1, seed=2))
            #             cell_bw = tf.contrib.rnn.LSTMCell(64, state_is_tuple=True,  initializer=tf.truncated_normal_initializer(-0.1, 0.1, seed=2))
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=keep_prob)

            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length,
                                                              dtype=tf.float32)
            output = tf.concat(outputs, 2)
            state = tf.concat(states, 2)

    return [output, state]


