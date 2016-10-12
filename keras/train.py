import time
from concept_extractor import *
from include.data import *
import yaml
from include.models import *
from keras.callbacks import EarlyStopping, ModelCheckpoint


def train(conf, video_model, X_train, Y_train, X_val, Y_val):
    # some callbacks
    fpath = './model/' + conf['train_name'] + '_weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
    cp_cb = ModelCheckpoint(filepath=fpath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    es_cb = EarlyStopping(monitor='val_loss', patience=conf['patience'], verbose=1, mode='auto')
    # tb_cb = TensorBoard(log_dir="./log", histogram_freq=1)
    # configuration
    sgd = SGD(lr=conf['lr'], decay=conf['decay'], momentum=conf['momentum'], nesterov=conf['nesterov'])
    video_model.compile(optimizer=sgd, loss=conf['loss_type']) #'categorical_crossentropy'

    # now fitting
    video_model.fit(X_train, Y_train,
                    batch_size=conf['batch_size'],
                    nb_epoch=conf['epochs'],
                    verbose=conf['verbose'],
                    validation_data=(X_val, Y_val),
                    shuffle=conf['shuffle'],
                    callbacks=[cp_cb, es_cb])#, tb_cb])
    return video_model


def load_split(filename):
    with h5py.File(filename, 'r') as f:
        X = f.get('X')[:]
        Y = f.get('Y')[:]
        files = f.get('files')[:]
    return X, Y, files


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Training a matcher")
    argparser.add_argument("conf", type=str, help="configuation file")
    args = argparser.parse_args()
    conf = yaml.load(open(args.conf, 'r'))

    # Loading doc2vec model
    d2v = doc2vec.Doc2Vec.load(conf['language_model'])

    # Loading split
    X_train, Y_train, train_files = load_split(conf['train'])
    X_val, Y_val, val_files = load_split(conf['val'])

    # Loading video model
    video_model = VideoTextModel(conf['in_dim'], conf['v_dim'], conf['model_name'])

    # Training (and Validating)
    print('============================== TRAINING ==============================')
    if conf['finetune'] is True:
        video_model.load_weights(conf['finetune_from'])
    video_model.summary()
    print("Train video list:")
    for file in train_files: print(file)
    print("Val video list:")
    for file in val_files: print(file)
    if conf['epochs'] > 0:
        train(conf, video_model, X_train, Y_train, X_val, Y_val)
        video_model.save_weights(conf['output_model'] + "_" + str(int(round(time.time() * 1000))))
    print('======================== FINISHED TRAINING ===========================')





