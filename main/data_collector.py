import json
import configure as conf
import numpy as np

if conf.post_encoder:
    from post_encoder_model import create_batch_month_vect
from collections import OrderedDict


def read_data(input_path, word_idx, if_increase_dict):
    seqs = []
    graphs = []
    if conf.post_encoder:
        g_posts = []

    if if_increase_dict:
        word_idx[conf.GO] = 1
        word_idx[conf.EOS] = 2
        word_idx[conf.unknown_word] = 3

    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            jo = json.loads(line, object_pairs_hook=OrderedDict)
            seq = jo['seq']
            seqs.append(seq)
            if if_increase_dict:
                for w in seq.split():
                    if w not in word_idx:
                        word_idx[w] = len(word_idx) + 1

                if conf.um_specific:
                    for time in range(0, conf.timestamp):
                        for id in jo['g_ids_features'][str(time)]:
                            # for id in jo['g_ids_features'][time]:
                            features = jo['g_ids_features'][str(time)][str(id)]
                            # features = jo['g_ids_features'][time][id]
                            for w in features.split():
                                if len(w) == 0:
                                    continue
                                if w not in word_idx:
                                    word_idx[w] = len(word_idx) + 1
                else:
                    for id in jo['g_ids_features']:
                        features = jo['g_ids_features'][id]
                        for w in features.split():
                            if w not in word_idx:
                                word_idx[w] = len(word_idx) + 1

            graph = {}
            graph['g_ids'] = jo['g_ids']  # Number of forums
            graph['g_ids_features'] = jo['g_ids_features']
            graph['g_adjseq'] = jo['g_adjseq']  # graph connection
            if conf.post_encoder:
                g_posts.append(jo['posts'])  # added by TC
                # g_posts.append(' '.join(jo['posts'].split(' ')[0:10])) # for bert model. Taking 10 words only
            graphs.append(graph)
        if conf.post_encoder:
            # batch_month_vect = model_enhancement.create_batch_month_vect(g_posts)
            batch_month_vect = create_batch_month_vect(g_posts)
            # pickle.dump( batch_month_vect, open( "batch_month_vect.pk", "wb" ) )
        else:
            batch_month_vect = None
    return seqs, graphs, batch_month_vect


def read_data_wo_post(input_path, word_idx, if_increase_dict):
    seqs = []
    graphs = []

    if if_increase_dict:
        word_idx[conf.GO] = 1
        word_idx[conf.EOS] = 2
        word_idx[conf.unknown_word] = 3

    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            jo = json.loads(line, object_pairs_hook=OrderedDict)
            seq = jo['seq']
            seqs.append(seq)
            if if_increase_dict:
                for w in seq.split():
                    if w not in word_idx:
                        word_idx[w] = len(word_idx) + 1

                if conf.um_specific:
                    for time in range(0, conf.timestamp):
                        for id in jo['g_ids_features'][str(time)]:
                            features = jo['g_ids_features'][str(time)][id]
                            for w in features.split():
                                if len(w) == 0:
                                    continue
                                if w not in word_idx:
                                    word_idx[w] = len(word_idx) + 1
                else:
                    for id in jo['g_ids_features']:
                        features = jo['g_ids_features'][id]
                        for w in features.split():
                            if w not in word_idx:
                                word_idx[w] = len(word_idx) + 1

            graph = {}
            graph['g_ids'] = jo['g_ids']
            graph['g_ids_features'] = jo['g_ids_features']
            graph['g_adjseq'] = jo['g_adjseq']
            graphs.append(graph)

    return seqs, graphs


def vectorize_data(word_idx, texts):
    tv = []
    for text in texts:
        stv = []
        for w in text.split():
            if w not in word_idx:
                stv.append(word_idx[conf.unknown_word])
            else:
                stv.append(word_idx[w])
        tv.append(stv)
    return tv


def cons_batch_graph(graphs):
    g_fw_adj_seq = []
    g_bw_adj_seq = []

    g_ids = {}
    g_ids_features = {}
    g_nodes = []
    g_id_gid_map = []

    for g in graphs:  # x , graph=96
        ids = g['g_ids']
        if conf.um_specific:
            features = {}  # [{0:['kw1t0 kw2t0...',...,'kw1t95 kw2t95...']....79:['kw1t0 kw2t0...',...,'kw1t95 kw2t95...']}]
            for time in range(0, conf.timestamp):
                for id in ids:
                    # id = int(id)
                    if id not in features:
                        features[id] = []
                    features[id].append(g['g_ids_features'][str(time)][str(id)])
                    # features[id].append(g['g_ids_features'][time][id])

        else:
            features = g['g_ids_features']  # Change here from line 78
        nodes = []
        id_gid_map = {}
        offset = len(g_ids.keys())
        for id in ids:
            id = int(id)
            g_ids[offset + id] = len(g_ids.keys())
            g_ids_features[offset + id] = features[str(id)]  # Change here
            id_gid_map[id] = offset + id
            nodes.append(offset + id)
        g_nodes.append(nodes)
        g_id_gid_map.append(id_gid_map)

    for time in range(0, conf.timestamp):
        g_fw_adj = {}
        g_bw_adj = {}
        for i, g in enumerate(graphs):
            id_adj = g['g_adjseq'][time]
            for id in id_adj:
                adj = id_adj[id]
                id = int(id)
                g_id = g_id_gid_map[i][id]
                if g_id not in g_fw_adj:
                    g_fw_adj[g_id] = []
                for t in adj:
                    t = int(t)
                    g_t = g_id_gid_map[i][t]
                    g_fw_adj[g_id].append(g_t)
                    if g_t not in g_bw_adj:
                        g_bw_adj[g_t] = []
                    g_bw_adj[g_t].append(g_id)

        node_size = len(g_ids.keys())
        for id in range(node_size):
            if id not in g_fw_adj:
                g_fw_adj[id] = []
            if id not in g_bw_adj:
                g_bw_adj[id] = []

        g_fw_adj_seq.append(g_fw_adj)
        g_bw_adj_seq.append(g_bw_adj)

    graph = {}
    graph['g_ids'] = g_ids
    graph['g_ids_features'] = g_ids_features
    graph['g_nodes'] = g_nodes
    graph['g_fw_adj_seq'] = g_fw_adj_seq
    graph['g_bw_adj_seq'] = g_bw_adj_seq

    return graph


def vectorize_batch_graph(graph, word_idx):
    # vectorize the graph feature and normalize the adj info
    id_features = graph['g_ids_features']
    gv = {}  # one snapshot graph
    nv = []  # nv = node vector

    #############################################################################
    if conf.node_feature_format == "bow":
        for id in graph['g_ids_features']:
            feature = graph['g_ids_features'][id]
            if conf.um_specific:
                fv = np.zeros((conf.timestamp, conf.word_vocab_size))  # 2D array
                for time in range(0, conf.timestamp):
                    for token in feature[time].split():
                        if len(token) == 0:
                            continue
                        if token in word_idx:
                            fv[time][word_idx[token]] = 1
                        else:
                            fv[time][word_idx[conf.unknown_word]] = 1
            else:
                fv = np.zeros(conf.word_vocab_size)
                for token in feature.split():
                    if len(token) == 0:
                        continue
                    if token in word_idx:
                        fv[word_idx[token]] = 1
                    else:
                        fv[word_idx[conf.unknown_word]] = 1
            nv.append(fv)
        gv['g_ids_features'] = np.array(nv)
    elif conf.node_feature_format == "word_emb":
        word_max_len = 0
        for id in id_features:
            feature = id_features[id]
            word_max_len = max(word_max_len, len(feature.split()))

        for id in graph['g_ids_features']:
            feature = graph['g_ids_features'][id]
            fv = []
            for token in feature.split():
                if len(token) == 0:
                    continue
                if token in word_idx:
                    fv.append(word_idx[token])
                else:
                    fv.append(word_idx[conf.unknown_word])

            for _ in range(word_max_len - len(fv)):
                fv.append(0)
            fv = fv[:word_max_len]
            nv.append(fv)

        nv.append([0 for temp in range(word_max_len)])
        gv['g_ids_features'] = np.array(nv)
        ########################################################################

    degree_max_size = conf.sample_size_per_layer

    # construct forward vector
    g_fw_adj_seq_v = []
    for g_fw_adj in graph['g_fw_adj_seq']:
        g_fw_adj_v = []

        for id in sorted(g_fw_adj):
            adj = g_fw_adj[id]
            for _ in range(degree_max_size - len(adj)):
                adj.append(len(g_fw_adj.keys()))
            adj = adj[:degree_max_size]
            g_fw_adj_v.append(adj)

        # PAD node directs to the PAD node
        g_fw_adj_v.append([len(g_fw_adj.keys()) for _ in range(degree_max_size)])

        g_fw_adj_seq_v.append(g_fw_adj_v)

    # construct backward vector
    g_bw_adj_seq_v = []
    for g_bw_adj in graph['g_bw_adj_seq']:
        g_bw_adj_v = []

        for id in sorted(g_bw_adj):
            adj = g_bw_adj[id]
            for _ in range(degree_max_size - len(adj)):
                adj.append(len(g_bw_adj.keys()))
            adj = adj[:degree_max_size]
            g_bw_adj_v.append(adj)

        # PAD node directs to the PAD node
        g_bw_adj_v.append([len(g_bw_adj.keys()) for _ in range(degree_max_size)])

        g_bw_adj_seq_v.append(g_bw_adj_v)

    gv['g_ids'] = graph['g_ids']
    gv['g_nodes'] = np.array(graph['g_nodes'])
    gv['g_bw_adj_seq'] = np.array(g_bw_adj_seq_v)
    gv['g_fw_adj_seq'] = np.array(g_fw_adj_seq_v)

    return gv
