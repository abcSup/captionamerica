import collections
import os
from os.path import join, exists
import json
import random
import pickle
import multiprocessing as mp

import h5py
import numpy as np
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences

def isFilename(word):
	try:
		return word.encode('ascii')
	except UnicodeEncodeError:
		return False

train_dir = '/mnt/Users/abcSup/Downloads/ai_challenger_caption_train_20170902/caption_train_images_20170902/'
val_dir =  ''
data_dir = './data/'
train_token_fname = join(data_dir, 'train_token.txt')
val_token_fname = join(data_dir, 'val_token.txt')
wtoi_fname = join(data_dir, 'wtoi.p')
itow_fname = join(data_dir, 'itow.p')

token_fname = train_token_fname
img_dir = train_dir
output_h5 = 'train_data.h5'

vocab_size = 10000
seq_len = 39 + 2

def read_dataset():
    words = []
    img_to_caps = {}
    longest_s = 0

    with open(token_fname, encoding="utf-8") as file:
        for l in file:
            if isFilename(l):
                temp_imgname = l.replace(' ', '').strip()
                img_to_caps[temp_imgname] = []
            else:
                words.extend(l.split())
                img_to_caps[temp_imgname].append(l)

                len_s = len(l.split())
                longest_s = len_s if longest_s < len_s else longest_s

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
            seq = [wtoi[w] for w in caption.split() if w in wtoi]
            seq.insert(0, wtoi['begin'])
            seq.insert(len(seq), wtoi['end'])
            seqs.append(seq)
        img_to_seqs[img_name] = seqs

    return img_to_seqs

def compute_caption(img_name):
    caps = []
    caps_len = []
    num_captions = len(img_to_seqs[img_name])
    assert num_captions > 0
    caps = pad_sequences(img_to_seqs[img_name], maxlen=seq_len, padding='post')

    return caps, num_captions
    
def prepare_datasets(img_to_seqs):
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
    caps = f.create_dataset("caps", (M, seq_len), dtype='uint32')
    caps_start_idx = f.create_dataset("caps_start_idx", (N,), dtype='uint32')
    caps_end_idx = f.create_dataset("caps_end_idx", (N,), dtype='uint32')

    result = pool.imap(compute_caption, [i for i in img_names])
            
    counter = 0
    for i, data in enumerate(tqdm(result)):
        caps[counter:counter+data[1],:] = (data[0])
        caps_start_idx[i] = counter
        caps_end_idx[i] = counter + data[1] - 1
        counter += data[1]

    print("Done.")
    f.close()

    pool.close()
    pool.join()

if __name__ == "__main__":
    words, img_to_caps = read_dataset()

    if exists(wtoi_fname) and exists(itow_fname):
        print("Load vocab")
        wtoi = pickle.load(open(wtoi_fname, 'rb'))
        itow = pickle.load(open(itow_fname, 'rb'))
    else:
        wtoi, itow = build_vocab(words, vocab_size)
        print("Create and pickle vocab")
        pickle.dump(wtoi, open(wtoi_fname), 'wb')
        pickle.dump(itow, open(itow_fname), 'wb')
    
    img_to_seqs = encode_captions(img_to_caps, wtoi)
    del words, img_to_caps
    
    prepare_datasets(img_to_seqs)
