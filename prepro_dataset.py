import collections
import os
from os.path import join, exists
import json
import random
import pickle
import multiprocessing as mp

import h5py
import jieba
import numpy as np
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences

from configuration import Config

train_dir = '/mnt/Users/abcSup/Downloads/ai_challenger_caption_train_20170902/caption_train_images_20170902/'
annotation_file = '/mnt/Users/abcSup/Downloads/ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json'
data_dir = './data/'
wtoi_fname = join(data_dir, 'wtoi.p')
itow_fname = join(data_dir, 'itow.p')

img_dir = train_dir
output_h5 = 'train_data.h5'

def tokenize(img):
    img_id = img['image_id']
    words, caps, length = [], [], []
    for c in img['caption']:
        cap = list(jieba.cut(c, cut_all=False))
        words.extend(cap)
        caps.append(cap)
        length.append(len(cap))

    return img_id, caps, words, length

def read_dataset():
    pool = mp.Pool(mp.cpu_count())

    words, length = [], []
    img_to_caps = {}

    with open(annotation_file, encoding="utf-8") as file:
        data = json.load(file)

    result = pool.imap(tokenize, [img for img in data])

    for r in tqdm(result):
        img_to_caps[r[0]] = r[1]
        words.extend(r[2])
        length.extend(r[3])

    longest_s = max(length)

    print('No. of images ', len(img_to_caps))
    print('No. of words ', len(words))
    print('No. of vocab ', len(collections.Counter(words)))
    print('Longest captions ', longest_s)

    return words, img_to_caps


def build_vocab(words, vocab_size):
    """Create vocabulary"""
    count = [['empty', -1], ['begin', -1], ['end', -1], ['unk', -1]]
    count.extend(collections.Counter(words).most_common(vocab_size - 4))

    wtoi = dict()
    for word, _ in count:
        wtoi[word] = len(wtoi)
    itow = dict(zip(wtoi.values(), wtoi.keys()))

    return wtoi, itow

def encode_captions(img_to_caps, wtoi):
    img_to_seqs = {}
    for img_name, captions in img_to_caps.items():
        seqs = []
        for caption in captions:
            seq = [wtoi[w] for w in caption if w in wtoi]
            seq.insert(0, wtoi['begin'])
            seq.insert(len(seq), wtoi['end'])
            seqs.append(seq)
        img_to_seqs[img_name] = seqs

    return img_to_seqs

def compute_caption(data):
    # forgive me, DRY God
    config = Config()

    img_name, captions = data

    num_captions = len(captions)
    assert num_captions > 0
    caps = pad_sequences(captions, maxlen=config.cap_len, padding='post')

    return caps, num_captions
    
def prepare_datasets(img_to_seqs, cap_len):
    pool = mp.Pool(mp.cpu_count())

    img_names = [x for x in img_to_seqs.keys()]
    random.seed(123)
    random.shuffle(img_names)
    pickle.dump(img_names, open(join(data_dir, 'img_names.p'), 'wb'))

    N = len(img_names)
    M = sum([len(img_to_seqs[i]) for i in img_names])
    print("Number of Images: ", N)
    print("Number of Captions: ", M)

    print("Creating hdf5 file")
    f = h5py.File(os.path.join(data_dir, output_h5), 'w')
    caps = f.create_dataset("caps", (M, cap_len), dtype='uint32')
    caps_start_idx = f.create_dataset("caps_start_idx", (N,), dtype='uint32')
    caps_end_idx = f.create_dataset("caps_end_idx", (N,), dtype='uint32')

    result = pool.imap(compute_caption, [(i, img_to_seqs[i]) for i in img_names])
            
    counter = 0
    for i, data in enumerate(tqdm(result)):
        caps[counter:counter+data[1],:] = (data[0])
        caps_start_idx[i] = counter
        caps_end_idx[i] = counter + data[1] - 1
        counter += data[1]

    f.close()

    pool.close()
    pool.join()

def main():
    config = Config()

    print("Read dataset")
    words, img_to_caps = read_dataset()

    print("Create and pickle vocab")
    wtoi, itow = build_vocab(words, config.vocab_size)
    pickle.dump(wtoi, open(wtoi_fname, 'wb'))
    pickle.dump(itow, open(itow_fname, 'wb'))
    
    img_to_seqs = encode_captions(img_to_caps, wtoi)
    del words, img_to_caps
    
    prepare_datasets(img_to_seqs, config.cap_len)
    print("Done")

if __name__ == "__main__":
    main()
