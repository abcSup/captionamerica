from os.path import join, exists
import pickle
import random
import argparse

import numpy as np
import h5py
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt

from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.applications.inception_v3 import preprocess_input
from keras.utils.data_utils import Sequence

from configuration import Config

train_imgdir = '/mnt/Users/abcSup/Downloads/ai_challenger_caption_train_20170902/caption_train_images_20170902/'
val_imgdir = '/mnt/Users/abcSup/Downloads/ai_challenger_caption_validation_20170910/caption_validation_images_20170910/'

class DataGenerator(object):

    def __init__(self, config, dataset):
        if dataset == "train":
            self.data_dir = './data/train'
            self.img_dir = train_imgdir
        elif dataset == "val":
            self.data_dir = './data/val'
            self.img_dir = val_imgdir
        else:
            raise Exception('Wrong dataset!')

        self.wtoi = pickle.load(open(join(self.data_dir, 'wtoi.p'), 'rb'))
        self.itow = pickle.load(open(join(self.data_dir, 'itow.p'), 'rb'))
        self.imglist_file = join(self.data_dir, 'img_names.p')

        h5file = join(self.data_dir, 'train_data.h5') 
        self.data = h5py.File(h5file, 'r')

        img_names = pickle.load(open(self.imglist_file, 'rb'))
        self.img_names = np.array(img_names)

        self.vocab_size = config.vocab_size
        self.seq_len = config.seq_len
        self.batch_size = config.batch_size
        self.num_batch = config.num_batch

    def seq_to_cap(self, seq):
        remove = [0, 1, 2]
        cap = [self.itow[w] for w in seq if w not in remove]
        return ''.join(cap)
    
    def return_img(self, img_name):
        path = join(self.img_dir, img_name)
        return imread(path)

    def return_groundtruth(self, img_name):
        idx = np.where(self.img_names == img_name)[0][0]
        start = self.data['caps_start_idx'][idx]
        end = self.data['caps_end_idx'][idx]
        caps = self.data['caps'][range(start,end+1)]

        return caps

    def _preprocess_caps(self, caps, seq_len):
        input_seqs, target_seqs = [], []
        for seq in caps:
            input_seqs.append(seq[:-1])
            target_seqs.append(seq[1:])

        input_seqs = np.asarray(input_seqs)
        input_seqs[input_seqs == self.wtoi['end']] = 0
        target_seqs = np.asarray(target_seqs)

        sample_weights = np.zeros([len(target_seqs), self.seq_len], dtype=np.bool)
        sample_weights[target_seqs>0] = 1

        target_seqs = to_categorical(target_seqs, self.vocab_size)
        target_seqs = np.reshape(target_seqs, (len(input_seqs), self.seq_len, self.vocab_size))

        return input_seqs, target_seqs, sample_weights

    def _preprocess_imgs(self, img_names):
        """Preprocess images for InceptionV3"""
        images = []
        for i, img_name in enumerate(img_names):
            path = join(self.img_dir, img_name)
            try:
                img = imresize(imread(path), (299,299)).astype(np.float32)
                img = preprocess_input(img)
            except:
                print("GGWP for image: %s" % img_name)
                continue
            images.append(img)
        
        return images

    def generator(self):
        batch_size = self.batch_size
        num_batch = self.num_batch

        captions = self.data['caps']
        starts = self.data['caps_start_idx']
        ends = self.data['caps_end_idx']

        img_names = self.img_names

        while(True):           
            idxs = np.arange(len(img_names))
            np.random.shuffle(idxs)

            for i in range(num_batch):
                batch_idxs = idxs[i*batch_size:(i+1)*batch_size]
                batch_idxs = np.sort(batch_idxs).tolist()
                
                img_idxs = img_names[batch_idxs]
                images = self._preprocess_imgs(img_idxs)

                start_idxs = starts[batch_idxs]
                end_idxs = ends[batch_idxs]

                cap_idxs = []
                for x in range(batch_size):
                    cap_idxs.append(random.randint(start_idxs[x], end_idxs[x]))
                caps = captions[cap_idxs]
                
                input_seqs, output_seqs, weights  = self._preprocess_caps(caps, self.seq_len)

                yield ([images, input_seqs], output_seqs, weights)

    def test(self):
        for data in self.generator():
            images = data[0][0]
            input_seqs = data[0][1]
            output_seqs = data[1]
            weights = data[2]

            for i, img in enumerate(images):
                for w in input_seqs[i]:
                    print(self.itow[w], end='')
                print()
                for x in output_seqs[i]:
                    w = np.argmax(x)
                    print(self.itow[w], end='')
                print()
                for x in output_seqs[i][weights[i]]:
                    w = np.argmax(x)
                    print(self.itow[w], end='')
                print()
                plt.imshow(img)
                plt.show()

class CustomSequence(Sequence):

    def __init__(self, data, batch_size):
        self.data_dir = './data/'
        self.img_dir = '/mnt/Users/abcSup/Downloads/ai_challenger_caption_train_20170902/caption_train_images_20170902/'

        self.captions = data['caps']
        self.starts = data['caps_start_idx']
        self.ends = data['caps_end_idx']

        self.imglist_file = join(self.data_dir, 'img_names.p')
        img_names = pickle.load(open(self.imglist_file, 'rb'))
        self.img_names = np.array(img_names)

        self.vocab_size = 10000
        self.seq_len = 41 - 1
        self.batch_size = batch_size

        self.idxs = np.arange(len(img_names))
        np.random.shuffle(self.idxs)

    def _preprocess_caps(self, caps, seq_len):
        input_seqs, target_seqs = [], []
        for seq in caps:
            input_seqs.append(seq[:-1])
            target_seqs.append(seq[1:])

        input_seqs = np.asarray(input_seqs)
        target_seqs = np.asarray(target_seqs)

        sample_weights = np.zeros([len(target_seqs), self.seq_len])
        sample_weights[target_seqs>0] = 1

        target_seqs = to_categorical(target_seqs, self.vocab_size)
        target_seqs = np.reshape(target_seqs, (len(input_seqs), self.seq_len, self.vocab_size))

        return input_seqs, target_seqs, sample_weights

    def _preprocess_imgs(self, img_names):
        """Preprocess image for InceptionV3"""
        images = np.zeros((self.batch_size,299,299,3),dtype=np.float32)
        for i, img_name in enumerate(img_names):
            path = join(self.img_dir, img_name)
            try:
                img = imresize(imread(path), (299,299)).astype(np.float32)
                img = preprocess_input(img)
            except:
                print("GGWP for image: %s" % img_name)
                continue
            images[i] = img
        
        return images
    
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        print("Batch #%d" % idx)
        batch_idxs = self.idxs[idx*self.batch_size : (idx+1)*self.batch_size]
        batch_idxs = np.sort(batch_idxs).tolist()
        print("From %d to %d" % (idx*self.batch_size, (idx+1)*self.batch_size))

        img_idxs = self.img_names[batch_idxs]
        images = self._preprocess_imgs(img_idxs)

        start_idxs = self.starts[batch_idxs]
        end_idxs = self.ends[batch_idxs]

        cap_idxs = []
        for x in range(batch_size):
            cap_idxs.append(random.randint(start_idxs[x], end_idxs[x]))
        caps = self.captions[cap_idxs]
        
        input_seqs, output_seqs, weights  = self._preprocess_caps(caps, self.seq_len)

        return ([images, input_seqs], output_seqs, weights)

    def on_epoch_end(self):
        np.random.shuffle(self.idxs)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing DataGenerator')
    parser.add_argument("dataset",
                        help="The dataset to generate (train|val)")
    args = parser.parse_args()
    config = Config()

    data = DataGenerator(config, args.dataset)
    data.test()
