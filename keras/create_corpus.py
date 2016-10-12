from include.data import *
import yaml
import argparse


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Build a video captions corpus from MSRVTT or YT2T captions")
    argparser.add_argument("conf", type=str, help="configuration file")
    args = argparser.parse_args()
    conf = yaml.load(open(args.conf, 'r'))

    # load captions objects
    captions = []
    for i in range(0, conf['num_objects']):
        caption = {}
        caption['captions'] = []
        captions.append(caption)
    for cap in conf['captions']:
        load_y2t_captions(cap, captions)

    with open(conf['output_corpus'], 'w') as f:
        for caption in captions:
            for cap in caption['captions']:
                f.write((cap + '\n').encode('utf8'))
    f.close()
