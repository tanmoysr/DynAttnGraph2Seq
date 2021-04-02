import configure as conf
import data_collector
from model import Graph2SeqNN
if conf.post_encoder:
    print('Running with post encoder')
    import post_encoder_model as model_enhancement # Post Encoder
import loaderAndwriter as disk_helper
import numpy as np
import tensorflow as tf
import helpers as helpers
import datetime
import text_decoder
from evaluator import evaluate
import os
import argparse
import json

def main(mode):

    word_idx = {}

    if mode == "train":
        epochs = conf.epochs
        train_batch_size = conf.train_batch_size # train_batch_size = 50

        # read the training data from a file
        print("reading training data into the mem ...")
        # texts_train (treatment seq of all users seqs) =['seqUsr1',......'seqUsrN'],
        # graphs_train ={{usergraph1}....{usergraphN}}
        texts_train, graphs_train, posts_train = data_collector.read_data(conf.train_data_path, word_idx,
                                                                          if_increase_dict=True)

        print("reading development data into the mem ...")
        texts_dev, graphs_dev, posts_dev = data_collector.read_data(conf.dev_data_path, word_idx,
                                                                    if_increase_dict=False)
        print("writing word-idx mapping ...")
        disk_helper.write_word_idx(word_idx, conf.word_idx_file_path)

        print("vectoring training data ...")
        # tv_train = list of values of word_idx[keyword] [1, 2, 3, ......n] (n= number of keywords used), This is basically giving the list of appeared keywords
        tv_train = data_collector.vectorize_data(word_idx, texts_train)

        print("vectoring dev data ...")
        tv_dev = data_collector.vectorize_data(word_idx, texts_dev)

        conf.word_vocab_size = len(word_idx.keys()) + 1

        with tf.Graph().as_default():
            with tf.Session() as sess:
                model = Graph2SeqNN("train", conf, path_embed_method="lstm")

                model._build_graph()
                saver = tf.train.Saver(max_to_keep=None)
                sess.run(tf.initialize_all_variables())

                def train_step(seqs, decoder_seq_length, loss_weights, batch_graph, if_pred_on_dev=False):
                    dict = {}
                    dict['seq'] = seqs
                    dict['batch_graph'] = batch_graph
                    dict['loss_weights'] = loss_weights
                    dict['decoder_seq_length'] = decoder_seq_length

                    if not if_pred_on_dev:
                        _, loss_op, l21_loss = model.act(sess, "train", dict, if_pred_on_dev)
                        return loss_op, l21_loss
                    else:
                        sample_id, infer_summary, node_attention, feature_weights = model.act(sess, "train", dict, if_pred_on_dev)
                        return sample_id, infer_summary

                best_score_on_dev = 0.0
                for t in range(1, epochs + 1):
                    n_train = len(texts_train) # texts_train (treatment seq of all users seqs) =['seqUsr1',......'seqUsrN']
                    temp_order = list(range(n_train)) # users order
                    np.random.shuffle(temp_order)

                    if conf.post_encoder:
                        if conf.post_encoder_type =='bilstm':
                            encoded_batch_prestate_train = model_enhancement.encoded_batch_prestatefn(sess, posts_train,train_batch_size, model) # Post Encoder
                        elif conf.post_encoder_type == 'bert':
                            encoded_batch_prestate_train = model_enhancement.bert_encoding(sess, model, posts_train)

                    loss_sum = 0.0
                    l21_loss_sum = 0.0
                    for start in range(0, n_train, train_batch_size): # o:user_numbers:train_batch_size(50)
                        end = min(start+train_batch_size, n_train)
                        tv = []
                        graphs = []
                        for _ in range(start, end): # _ for throwaway variables. It just indicates that the loop variable isn't actually used.
                            idx = temp_order[_] # chose specific user from users order list
                            tv.append(tv_train[idx]) # tv_train = list of values of word_idx[keyword] [1, 2, 3, ......n] (n= number of keywords used), This is basically giving the list of appeared keywords
                            graphs.append(graphs_train[idx]) #idx = user; graphs =[{usergraph1}....{usergraphN}]

                        # batch graph = {'g_ids': {total keys = number of subforum* number of users} , 'g_ids_features': {userspecific subform keywords. total keys = number of subforum* number of users},
                        #     'g_nodes': [[0~79], [80~159] increamentally upto the N users], 'g_fw_adj_seq': [], 'g_bw_adj_seq': []}
                        batch_graph = data_collector.cons_batch_graph(graphs)
                        gv = data_collector.vectorize_batch_graph(batch_graph, word_idx)
                        if conf.post_encoder:
                            gv['g_posts'] = encoded_batch_prestate_train[start:end]

                        tv, tv_real_len, loss_weights = helpers.batch(tv)
                        print(start)
                        loss_op, l21_loss = train_step(tv, tv_real_len, loss_weights, gv)
                        loss_sum += loss_op
                        l21_loss_sum += l21_loss
                    #################### test the model on the dev data #########################
                    n_dev = len(texts_dev)
                    dev_batch_size = conf.dev_batch_size

                    # encoded_batch_prestate_dev=encoded_batch_prestate_train
                    if conf.post_encoder:
                        if conf.post_encoder_type == 'bilstm':
                            encoded_batch_prestate_dev = model_enhancement.encoded_batch_prestatefn(sess, posts_dev,
                                                                                            dev_batch_size, model)
                        elif conf.post_encoder_type == 'bert':
                            encoded_batch_prestate_dev = model_enhancement.bert_encoding(sess, model, posts_dev)

                    idx_word = {}
                    for w in word_idx:
                        idx_word[word_idx[w]] = w

                    pred_texts = []

                    for start in range(0, n_dev, dev_batch_size):
                        end = min(start+dev_batch_size, n_dev)
                        tv = []
                        graphs = []
                        for _ in range(start, end):
                            tv.append(tv_dev[_])
                            graphs.append(graphs_dev[_])

                        batch_graph = data_collector.cons_batch_graph(graphs)
                        gv = data_collector.vectorize_batch_graph(batch_graph, word_idx)
                        if conf.post_encoder:
                            gv['g_posts'] = encoded_batch_prestate_dev[start: end]

                        tv, tv_real_len, loss_weights = helpers.batch(tv)

                        sample_id, infer_summary = train_step(tv, tv_real_len, loss_weights, gv, if_pred_on_dev=True)

                        for tmp_id in sample_id:
                            pred_texts.append(text_decoder.decode_text(tmp_id, idx_word))

                    score = evaluate(type="rouge", golds=texts_dev, preds=pred_texts)
                    # score = evaluate(type="bleu", golds=texts_dev, preds=pred_texts)[-1]  # bleu [1,2,3,4]
                    if_save_model = False
                    if score >= best_score_on_dev:
                        best_score_on_dev = score
                        if_save_model = True

                    time_str = datetime.datetime.now().isoformat()
                    print('-----------------------')
                    print('time:{}'.format(time_str))
                    print('Epoch', t)
                    print('Score on Dev: {}'.format(score))
                    print('Best score on Dev: {}'.format(best_score_on_dev))
                    print('Loss on train:{}'.format(loss_sum))
                    print('L2,1 loss on train:{}'.format(l21_loss_sum))
                    if if_save_model:
                        save_path = "../saved_model/"
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)

                        path = saver.save(sess, save_path + 'model', global_step=0)
                        print("Already saved model to {}".format(path))

                    print('-----------------------')

    elif mode == "test":
        test_batch_size = conf.test_batch_size

        print("reading test data into the mem ...")
        texts_test, graphs_test, posts_test = data_collector.read_data(conf.test_data_path, word_idx,
                                                                       if_increase_dict=False)
        print("reading word idx mapping from file")
        word_idx = disk_helper.read_word_idx_from_file(conf.word_idx_file_path)

        idx_word = {}
        for w in word_idx:
            idx_word[word_idx[w]] = w

        print("vectoring test data ...")
        tv_test = data_collector.vectorize_data(word_idx, texts_test)

        conf.word_vocab_size = len(word_idx.keys()) + 1

        with tf.Graph().as_default():
            with tf.Session() as sess:
                model = Graph2SeqNN("test", conf, path_embed_method="lstm")
                model._build_graph()
                saver = tf.train.Saver(max_to_keep=None)


                model_path_name = "../saved_model/model-0"
                model_pred_path = "../saved_model/prediction.txt"
                model_feature_path = "../saved_model/feature_selection.txt"
                model_node_attention_path = "../saved_model/node_attention.txt"
                model_attention = "../saved_model/"

                writer = tf.summary.FileWriter(model_attention)
                saver.restore(sess, model_path_name)

                def test_step(seqs, decoder_seq_length, loss_weights, batch_graph):
                    dict = {}
                    dict['seq'] = seqs
                    dict['batch_graph'] = batch_graph
                    dict['loss_weights'] = loss_weights
                    dict['decoder_seq_length'] = decoder_seq_length
                    sample_id, infer_summary, node_attention, feature_weights = model.act(sess, "test", dict, if_pred_on_dev=False)
                    return sample_id, infer_summary, node_attention, feature_weights

                n_test = len(texts_test)

                pred_texts = []
                global_graphs = []

                tv = []
                graphs = []
                for _ in range(0, n_test):
                    tv.append(tv_test[_])
                    graphs.append(graphs_test[_])
                    global_graphs.append(graphs_test[_])

                if conf.post_encoder:
                    if conf.post_encoder_type=='bilstm':
                        encoded_batch_prestate_test = model_enhancement.encoded_batch_prestatefn(sess, posts_test,
                                                                                         test_batch_size,
                                                                                         model)
                    elif conf.post_encoder_type=='bert':
                        encoded_batch_prestate_test = model_enhancement.bert_encoding(sess, model, posts_test)
                batch_graph = data_collector.cons_batch_graph(graphs)
                gv = data_collector.vectorize_batch_graph(batch_graph, word_idx)
                if conf.post_encoder:
                    gv['g_posts'] = encoded_batch_prestate_test
                tv, tv_real_len, loss_weights = helpers.batch(tv)

                sample_id, infer_summary, node_attention, feature_weights = test_step(tv, tv_real_len, loss_weights, gv)

                for tem_id in sample_id:
                    pred_texts.append(text_decoder.decode_text(tem_id, idx_word))

                if infer_summary is not None:
                    writer.add_summary(infer_summary)
                writer.close()

                with open('../saved_model/gold.txt', 'w') as f:
                    for item in texts_test:
                        f.write("%s\n" % item)
                with open('../saved_model/pred.txt', 'w') as f:
                    for item in pred_texts:
                        f.write("%s\n" % item)

                acc = evaluate(type="acc", golds=texts_test, preds=pred_texts)
                print("acc on test set is {}".format(acc))

                # bscore = evaluate(type="bleu", golds=texts_test, preds=pred_texts)
                # print("bleu score on test set is {}".format(bscore))
                bscore1, bscore2, bscore3, bscore4 = evaluate(type="bleu", golds=texts_test, preds=pred_texts)
                print("bleu1 score on test set is {}".format(bscore1))
                print("bleu2 score on test set is {}".format(bscore2))
                print("bleu3 score on test set is {}".format(bscore3))
                print("bleu4 score on test set is {}".format(bscore4))

                rscore = evaluate(type="rouge", golds=texts_test, preds=pred_texts)
                print("rouge score on test set is {}".format(rscore))

                lscore = evaluate(type="levenshtein", golds=texts_test, preds=pred_texts)
                print("levenshtein distance on test set is {}".format(lscore))

                # write prediction result into a file
                with open(model_pred_path, 'w+') as f:
                    for _ in range(len(global_graphs)):
                        f.write("graph:\t"+json.dumps(global_graphs[_])+"\nGold:\t"+texts_test[_]+"\nPredicted:\t"+pred_texts[_]+"\n")
                        if texts_test[_].strip() ==  pred_texts[_].strip():
                            f.write("Correct\n\n")
                        else:
                            f.write("Incorrect\n\n")

                # write feature weights result into a file
                # feature_weights [2, feature, time,?]
                with open(model_feature_path, 'w+') as f:
                    for di in feature_weights:
                        for _ in range(len(di)):
                            f.write("feature id:" + str(_) + "weights:\n" +
                                    json.dumps(np.around(di[_], decimals=2).tolist()) + "\n")
                        f.write("==========================================\n")

                # write attention weights result into a file
                # node attention [time, batch_size, node_num]
                with open(model_node_attention_path, 'w+') as f:
                    for _ in range(len(node_attention)):
                        f.write("timestamp:" + str(_) + "attentions:\n" +
                                json.dumps(np.around(node_attention[_], decimals=2).tolist()) + "\n")
                    f.write("==========================================\n")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("mode", type=str, choices=["train", "test"])
    argparser.add_argument("-sample_size_per_layer", type=int, default=conf.sample_size_per_layer, help="sample size at each layer")
    argparser.add_argument("-sample_layer_size", type=int, default=conf.sample_layer_size, help="sample layer size")
    argparser.add_argument("-epochs", type=int, default=conf.epochs, help="training epochs")
    argparser.add_argument("-learning_rate", type=float, default=conf.learning_rate, help="learning rate")
    argparser.add_argument("-word_embedding_dim", type=int, default=conf.word_embedding_dim, help="word embedding dim")
    argparser.add_argument("-hidden_layer_dim", type=int, default=conf.hidden_layer_dim)

    config = argparser.parse_args()

    mode = config.mode
    conf.sample_layer_size = config.sample_layer_size
    conf.sample_size_per_layer = config.sample_size_per_layer
    conf.epochs = config.epochs
    conf.learning_rate = config.learning_rate
    conf.word_embedding_dim = config.word_embedding_dim
    conf.hidden_layer_dim = config.hidden_layer_dim

    main(mode)
