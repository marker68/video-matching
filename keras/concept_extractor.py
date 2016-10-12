import argparse
import yaml
from keras.optimizers import SGD
from include.models import *


def process(model, img_dir, output, synset):
    # Load images
    imgs = get_files(img_dir)
    n = len(imgs)
    k = n / 10
    labels = []
    for i in range(0, k + 1):
        print("loop " + str(i))
        st = i * 10
        ed = st + 10
        if ed >= n: ed = n
        ims = load_images(imgs[st:ed])
        label = classify(model, ims, synset)
        for l in label: labels.append(l)
    # Print some results
    print("First concepts:")
    print(labels[0])
    savelabels(list(set(labels)), output)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Extracting concepts using CNN")
    argparser.add_argument("conf", type=str, help="configuration file")
    args = argparser.parse_args()
    conf = yaml.load(open(args.conf, 'r'))

    # Test pretrained model
    model = load_model(conf['h5_file'], conf['type'])
    sgd = SGD(lr=conf['lr'], decay=conf['decay'], momentum=conf['momentum'], nesterov=conf['nesterov'])
    model.compile(optimizer=sgd, loss=conf['loss_type']) #'categorical_crossentropy'

    # process all videos
    for i in range(0,conf['num_videos']):
        print("Video " + str(i))
        img_dir = conf['img_dir'] + "/" + str(i+1) + "/rgb/"
        output = conf['output_dir'] + "/" + str(i+1) + ".txt"
        process(model, img_dir, output, conf['synset'])






