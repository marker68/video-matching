import argparse
from gensim.utils import *
from include.models import *
from include.data import *
import yaml


def load_data(conf, lm):
    # load video data and captions
    files = get_files(conf['video_dir'])
    V = load_all_data(files, conf['num_frames'], conf['crop_size'])
    # load captions
    if conf['cap_type'] == 0:
        captions = load_v2t_captions(conf['captions'])
    else:
        captions = []
        for i in range(0, len(files)):
            caption = {}
            caption['captions'] = []
            captions.append(caption)
        for cf in conf['captions']:
            load_y2t_captions(cf, captions)

    # compute caption vectors
    k = conf['n_captions']
    caption_vs = []
    for caption in captions:
        cs = caption['captions']
        if conf['pool_type'] == 'concat':
            cap_vec = []
            if len(cs) < k:
                lc = cs[len(cs)-1]
                for i in range(0,k-len(cs)): cs.append(lc)
            for c in cs[0:k]:
                caption_words = list(tokenize(c, lowercase=True, deacc=True))
                caption_vec = lm.infer_vector(caption_words)
                cap_vec = np.concatenate((cap_vec, caption_vec))
        else:
            cap_vec = np.zeros(conf['v_dim'])
            for c in cs:
                caption_words = list(tokenize(c, lowercase=True, deacc=True))
                caption_vec = lm.infer_vector(caption_words)
                cap_vec += caption_vec
            cap_vec = cap_vec / len(cs)
        caption_vs.append(cap_vec)
    return V, caption_vs, files


def load_data_for_train(conf, lm):
    # load video data and captions
    files = get_files(conf['video_dir'])
    ids = list(range(0, len(files)))
    if conf['shuffle'] == 1: random.shuffle(ids)
    train_files = []
    val_files = []
    for i in range(0,len(files)):
        if i < conf['num_train']:
            train_files.append(files[ids[i]])
        else:
            val_files.append(files[ids[i]])
    V_train = load_all_data(train_files, conf['num_frames'], conf['crop_size'])
    V_val = load_all_data(val_files, conf['num_frames'], conf['crop_size'])
    # load captions
    if conf['cap_type'] == 0:
        captions = load_v2t_captions(conf['captions'])
    else:
        captions = []
        for i in range(0, len(files)):
            caption = {}
            caption['captions'] = []
            captions.append(caption)
        for cf in conf['captions']:
            load_y2t_captions(cf, captions)

    # compute caption vectors
    train_caption_v = []
    val_caption_v = []
    k = conf['n_captions']
    for i in range(0, len(captions)):
        cs = captions[ids[i]]['captions']
        if conf['pool_type'] == 'concat':
            cap_vec = []
            if len(cs) < k:
                lc = cs[len(cs)-1]
                for i in range(0,k-len(cs)): cs.append(lc)
            for c in cs[0:k]:
                caption_words = list(tokenize(c, lowercase=True, deacc=True))
                caption_vec = lm.infer_vector(caption_words)
                cap_vec = np.concatenate((cap_vec, caption_vec))
        else:
            cap_vec = np.zeros(conf['v_dim'])
            for c in cs:
                caption_words = list(tokenize(c, lowercase=True, deacc=True))
                caption_vec = lm.infer_vector(caption_words)
                cap_vec += caption_vec
            cap_vec = cap_vec / len(cs)
        if i < conf['num_train']: train_caption_v.append(cap_vec)
        else: val_caption_v.append(cap_vec)

    return V_train, V_val, train_caption_v, val_caption_v, train_files, val_files


def save_split(X, Y, files, filename):
    with h5py.File(filename, 'w') as f:
        f['X'] = X
        f['Y'] = Y
        f['files'] = files


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Creating data")
    argparser.add_argument("conf", type=str, help="configuration file")
    args = argparser.parse_args()
    conf = yaml.load(open(args.conf, 'r'))
    lm = doc2vec.Doc2Vec.load(conf['language_model'])
    if conf['mode'] == 'test':
        X, Y, files = load_data(conf, lm)
        save_split(X, Y, files, conf['output_file'])
    else:
        X_train, X_val, Y_train, Y_val, train_files, val_files = load_data_for_train(conf, lm)
        save_split(X_train, Y_train, train_files,conf['output_file_train'])
        save_split(X_val, Y_val, val_files, conf['output_file_val'])
