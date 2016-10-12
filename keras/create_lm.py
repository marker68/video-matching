import argparse
import yaml
from include.models import *

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Training a doc2vec model")
    argparser.add_argument("conf", type=str, help="configuration file")
    args = argparser.parse_args()
    conf = yaml.load(open(args.conf, 'r'))
    # build models
    sentences = doc2vec.TaggedLineDocument(conf['sentences'])
    d2v_model = doc2vec_model(sentences, conf['epochs'], conf['v_dim'])
    d2v_model.save(conf['output'])
