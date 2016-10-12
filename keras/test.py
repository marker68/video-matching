import argparse
import yaml
from include.evaluation import *
from gensim.utils import *
from include.models import *
from include.data import *


def load_test_data(h5_file):
    with h5py.File(h5_file, 'r') as f:
        X = f.get('X')[:]
        files = f.get('files')[:]
    return X, files


def load_test_data_with_y(h5_file):
    with h5py.File(h5_file, 'r') as f:
        X = f.get('X')[:]
        Y = f.get('Y')[:]
        files = f.get('files')[:]
    return X, Y, files


def load_caption_to_y(caption_file, d2v):
    captions = load_v2t_captions(caption_file)
    Y = []
    for caption in captions:
        caption_words = list(tokenize(caption['captions'][0], lowercase=True, deacc=True))
        caption_vec = d2v.infer_vector(caption_words)
        c = caption_vec
        if conf['pool_type'] == 'concat':
            for i in range(1, conf['n_captions']):
                caption_vec = np.concatenate((caption_vec, c))
        Y.append(caption_vec)
    return Y


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Testing a matcher")
    argparser.add_argument("conf", type=str, help="configuration file")
    args = argparser.parse_args()
    conf = yaml.load(open(args.conf, 'r'))

    # load data
    if conf['compute_d2v'] is True:
        X_test, test_files = load_test_data(conf['video_data'])
        d2v = doc2vec.Doc2Vec.load(conf['language_model'])
        Y_test = load_caption_to_y(conf['test_captions'], d2v)
    else: X_test, Y_test, test_files = load_test_data_with_y(conf['video_data'])
    vm = VideoTextModel(conf['in_dim'], conf['v_dim'], conf['model_name'])
    vm.load_weights(conf['video_model'])
    # testing
    print("Test file list:")
    for file in test_files: print(file)
    if conf['gt'] is False:
        test_without_gt(vm, X_test, Y_test, conf['refine'], conf['output_ranks'])
    else:
        test_with_gt(vm, X_test, Y_test, conf['refine'])
