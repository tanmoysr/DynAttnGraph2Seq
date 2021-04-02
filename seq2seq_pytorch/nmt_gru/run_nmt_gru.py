import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import argparse
# Customized library
from evaluator import evaluate
import configure_gru as config
import data_processor_gru as data_processor
import model_gru as model_s2s
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
    print("Number of GPU {}".format(torch.cuda.device_count()))
    print("GPU type {}".format(torch.cuda.get_device_name(0)))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')
    torch.backends.cudnn.benchmark = False
    # torch.cuda.memory_summary(device=None, abbreviated=False)
    # print(torch.backends.cudnn.version())
else:
    device = torch.device("cpu")
    print("Running on the CPU")
# device = torch.device("cpu")

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(run_mode):
    start_load = time.perf_counter()
    # index language (wrd2idx, idx2wrd) using the class below
    vocab_src = open(config.vocab_src, encoding='UTF-8').read().strip().split('\n')
    vocab_tgt = open(config.vocab_tgt, encoding='UTF-8').read().strip().split('\n')
    src_lang = data_processor.LanguageIndex(vocab_src)
    targ_lang = data_processor.LanguageIndex(vocab_tgt)
    # Read data & Create tensor
    if run_mode == 'train':
        input_tensor_train,target_tensor_train = data_processor.create_tensor(config.train_src, config.train_tgt, src_lang, targ_lang)
        input_tensor_val, target_tensor_val = data_processor.create_tensor(config.dev_src, config.dev_tgt, src_lang, targ_lang)
        if config.post_encodder:
            posts_train = data_processor.read_post(config.train_post)
            posts_val = data_processor.read_post(config.dev_post)
    elif run_mode == 'test':
        input_tensor_test, target_tensor_test = data_processor.create_tensor(config.test_src, config.test_tgt, src_lang, targ_lang)
        if config.post_encodder:
            posts_test = data_processor.read_post(config.test_post)
    # Hyper-parameter
    EPOCHS = config.epoch
    BATCH_SIZE = config.batch_size
    if run_mode == 'train':
        BUFFER_SIZE = len(input_tensor_train)
    elif run_mode == 'test':
        BUFFER_SIZE = len(input_tensor_test)

    N_BATCH = BUFFER_SIZE//BATCH_SIZE
    dropout = config.dropout
    embedding_dim = config.embedding_dim
    units = config.units
    vocab_inp_size = len(src_lang.word2idx)
    vocab_tar_size = len(targ_lang.word2idx)

    # convert the data to tensors and pass to the Dataloader
    # to create an batch iterator
    if run_mode == 'train':
        train_dataset = data_processor.MyData(input_tensor_train, target_tensor_train,posts_train)
        val_dataset = data_processor.MyData(input_tensor_val, target_tensor_val, posts_val)
        del posts_train, posts_val
        dataset_train = DataLoader(train_dataset, batch_size = BATCH_SIZE, drop_last=True,shuffle=True)
        dataset_val = DataLoader(val_dataset, batch_size = BATCH_SIZE,drop_last=True,shuffle=True)

    elif run_mode == 'test':
        test_dataset = data_processor.MyData(input_tensor_test, target_tensor_test, posts_test)
        dataset_test = DataLoader(test_dataset, batch_size = BATCH_SIZE,drop_last=True,shuffle=True)
        del posts_test
    print('Time taken for data loading {} sec\n'.format(time.perf_counter() - start_load))
    # define model
    encoder = model_s2s.Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE, dropout, config.post_encodder).to(device)
    decoder = model_s2s.Decoder(vocab_tar_size, embedding_dim, units, units, BATCH_SIZE).to(device)
    model = model_s2s.Seq2Seq(encoder, decoder, targ_lang, BATCH_SIZE, device).to(device)

    if run_mode == 'train':
        dev_score = 0
        model.apply(init_weights)
        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                               lr=config.learning_rate)
        print(f'The model has {count_parameters(model):,} trainable parameters')
        start = time.perf_counter()
        for epoch in range(EPOCHS):

            model.train()
            train_loss = 0
            for (batch, (inp, targ, inp_len, posts)) in enumerate(dataset_train):
                # sort the batch first to be able to use with pac_pack_sequence
                xs, ys, lens, posts_srt = data_processor.sort_batch(inp, targ, inp_len, posts)
                outputs, loss = model(xs.to(device), ys.to(device), lens.to(device), posts_srt.float().to(device))
                batch_loss = (loss / int(ys.size(1)))
                train_loss += batch_loss
                optimizer.zero_grad()
                loss.backward()
                ### UPDATE MODEL PARAMETERS
                optimizer.step()

                if batch % 5 == 0:  # %100
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                                 batch,
                                                                 batch_loss.detach().item()))

            gold_val=[]
            pred_val=[]
            dev_loss=0
            for (batch, (inp, targ, inp_len, posts)) in enumerate(dataset_val):
                # print(batch)
                # sort the batch first to be able to use with pac_pack_sequence
                xs, ys, lens, posts_srt = data_processor.sort_batch(inp, targ, inp_len, posts)
                outputs, loss = model(xs.to(device), ys.to(device), lens.to(device), posts_srt.float().to(device))
                batch_loss = (loss / int(ys.size(1)))
                dev_loss += batch_loss
                gold_val += data_processor.gold_seq_extraction(ys, targ_lang)
                pred_val += data_processor.pred_seq_extraction(outputs, targ_lang)

            print('Dev set Loss {:.4f}'.format(dev_loss / N_BATCH))
            score = evaluate(type="rouge", golds=gold_val, preds=pred_val)
            if score>=dev_score:
                dev_score=score
                # Save the model checkpoint
                torch.save(model.state_dict(), config.save_model_path + 'model_NMT.ckpt')
                print('Model saved')

            print("score on dev set is {}".format(score))
            print("best score on dev set is {}".format(dev_score))
            print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                                train_loss / N_BATCH))
            print('---------------------------------------------')
        print('Time taken for {} epochs {} sec\n'.format(epoch+1, time.perf_counter() - start))

    elif run_mode == 'test':
        start = time.perf_counter()
        checkpoint = torch.load(config.save_model_path + 'model_NMT.ckpt')
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        print('Model Loaded')
        total_loss = 0
        gold = []
        pred = []
        for (batch, (inp, targ, inp_len, posts)) in enumerate(dataset_test):
            # print(batch)
            # sort the batch first to be able to use with pac_pack_sequence
            xs, ys, lens, posts_srt = data_processor.sort_batch(inp, targ, inp_len, posts)
            outputs, loss = model(xs.to(device), ys.to(device), lens.to(device), posts_srt.float().to(device))
            batch_loss = (loss / int(ys.size(1)))
            total_loss += batch_loss
            gold+=data_processor.gold_seq_extraction(ys, targ_lang)
            pred+=data_processor.pred_seq_extraction(outputs, targ_lang)

        # write prediction result into a file
        with open(config.save_model_path+'gold_NMT.txt', 'w') as f:
            for item in gold:
                f.write("%s\n" % item)
        # write prediction result into a file
        with open(config.save_model_path+'pred_NMT.txt', 'w') as f:
            for item in pred:
                f.write("%s\n" % item)
        print('Test set Loss {:.4f}'.format(total_loss / N_BATCH))
        acc = evaluate(type="acc", golds=gold, preds=pred)
        print("acc on test set is {}".format(acc))
        bscore1, bscore2, bscore3, bscore4 = evaluate(type="bleu", golds=gold, preds=pred)
        print("bleu1 score on test set is {}".format(bscore1))
        print("bleu2 score on test set is {}".format(bscore2))
        print("bleu3 score on test set is {}".format(bscore3))
        print("bleu4 score on test set is {}".format(bscore4))

        rscore = evaluate(type="rouge", golds=gold, preds=pred)
        print("rouge score on test set is {}".format(rscore))

        lscore = evaluate(type="levenshtein", golds=gold, preds=pred)
        print("levenshtein distance on test set is {}".format(lscore))
        print('Time taken for test set {} sec\n'.format(time.perf_counter() - start))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("mode", type=str, choices=["train", "test"])
    terminal_parser = argparser.parse_args()
    run_mode = terminal_parser.mode
    main(run_mode)