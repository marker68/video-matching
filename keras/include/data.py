import h5py
import cv2
import numpy as np
import json
from os import listdir
from os.path import *
from gensim.models import *


def get_files(dir):
    files = [join(dir, f) for f in listdir(dir) if isfile(join(dir, f))]
    print("We have " + str(len(files)) + " objects")
    return files


def load_video(filename, num_frames=20, size=128):
    print(filename)
    cap = cv2.VideoCapture(filename)
    ret, frame = cap.read()
    tmp = cv2.resize(frame, (size,size))
    video = tmp
    while(True):
        ret, frame = cap.read()
        if frame is None: break
        tmp = cv2.resize(frame, (size,size))
        video = np.vstack([video,tmp])
    cap.release()
    n = len(video)
    video = np.reshape(video, (n/size,size,size,3))
    n /= size
    if n <= num_frames: return video
    k = n / num_frames
    r = n - k * num_frames
    v = video[r]
    for i in range(1,num_frames):
        v = np.vstack([v,video[i*k+r]])
    n = len(v)/size
    v = np.reshape(v,(n,size,size,3))
    return v


def load_all_data(files, num_frames=20, size=128):
    vs = np.zeros((len(files),num_frames,size,size,3))
    count = 0
    for file in files:
        v = load_video(file,num_frames,size)
        vs[count] = v
        count += 1
    return vs


def save_hdf5(filename, v, captions):
    with h5py.File(filename, 'w') as f:
        f['data'] = v
        f['captions'] = captions


def load_v2t_captions(filename):
    with open(filename) as f:
        content = f.readlines()

    captions = []
    for line in content:
        line = line[3:]
        line = line.lstrip()
        caption = {}
        caption['captions'] = [line]
        captions.append(caption)
    return captions


def load_y2t_captions(filename, caps):
    captions = json.load(open(filename, 'r'))['captions']
    for caption in captions:
        caps[caption['video_id']-1]['captions'].append(caption['caption'])


def doc2vec_model(sentences,n_epochs=10, dim=300):
    model = doc2vec.Doc2Vec(size=dim,alpha=0.025, min_alpha=0.025)
    model.build_vocab(sentences)
    for epoch in range(n_epochs):
        model.train(sentences)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay
    return model

