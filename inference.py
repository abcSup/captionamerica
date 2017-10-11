import argparse
import json
import random

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from keras.preprocessing import sequence
from keras.applications.inception_v3 import preprocess_input

from configuration import Config
from model import get_model
from data_generator import DataGenerator

def beamsearch(img, maxsample, k,
               use_unk=False, oov=3, empty=0, eos=2):
    """return k samples (beams) and their NLL scores, each sample is a sequence of labels,
    all samples starts with an `empty` label and end with `eos` or truncated to length of `maxsample`.
    You need to supply `predict` which returns the label probability of each sample.
    `use_unk` allow usage of `oov` (out-of-vocabulary) label in samples
    """
	
    dead_k = 0 # samples that reached eos
    dead_samples = []
    dead_scores = []
    live_k = 1 # samples that did not yet reached eos
    live_samples = [[1]]
    live_scores = [0]

    count = 0
    while live_k and dead_k < k:
        # for every possible live sample calc prob for every possible label 
        images = np.repeat(img, repeats=live_k, axis=0)
        padded_live_samples = sequence.pad_sequences(live_samples, maxlen=maxsample, padding='post')
        probs = model.predict([images, padded_live_samples], verbose= 0)[:,count,:]
        count += 1

        # total score for every sample is sum of -log of word prb
        cand_scores = np.array(live_scores)[:,None] - np.log(probs)
        if not use_unk and oov is not None:
        	cand_scores[:, oov] = 1e20
        cand_flat = cand_scores.flatten()

        # find the best (lowest) scores we have from all possible samples and new words
        ranks_flat = cand_flat.argsort()[:(k-dead_k)]
        live_scores = cand_flat[ranks_flat]

        # append the new words to their appropriate live sample
        voc_size = probs.shape[1]
        live_samples = [live_samples[r//voc_size]+[r%voc_size] for r in ranks_flat]

        # live samples that should be dead are...
        zombie = [s[-1] == eos or len(s) >= maxsample for s in live_samples]
		
        # add zombies to the dead
        dead_samples += [s for s,z in zip(live_samples,zombie) if z]  # remove first label == empty
        dead_scores += [s for s,z in zip(live_scores,zombie) if z]
        dead_k = len(dead_samples)
        # remove zombies from the living 
        live_samples = [s for s,z in zip(live_samples,zombie) if not z]
        live_scores = [s for s,z in zip(live_scores,zombie) if not z]
        live_k = len(live_samples)

    return dead_samples + live_samples, dead_scores + live_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("weight",
                        help="the weight file path")
    parser.add_argument("--reference",
                        help="output result according to reference file")

    args = parser.parse_args()
    config = Config()

    model = get_model(config)
    model.load_weights(args.weight)

    data = DataGenerator(config, 'val')

    if args.reference:
        with open(args.reference, encoding='utf-8') as file:
            ref = json.load(file)

        img_names = []
        for img in ref['images']:
            if img['file_name'] not in img_names:
                img_names.append(img['file_name'])

        output = []
        for i in tqdm(img_names):
            img = data._preprocess_imgs([i + '.jpg'])
            seqs, scores = beamsearch(img, config.seq_len, config.beam_size)
            seq = seqs[np.argmax(scores)]

            x = {'image_id': i.rstrip('.jpg'), 'caption': data.seq_to_cap(seq)}
            output.append(x)

        json.dump(output, open('./data/result.json', 'w'),
                  sort_keys=True, indent=2)
    else:
        img_names = data.img_names

        while True:
            img_name = random.choice(img_names)
            img = data._preprocess_imgs([img_name])
            seqs, scores = beamsearch(img, config.seq_len, config.beam_size)

            print("Prediction, scores")
            for i, seq in enumerate(seqs):
                print(data.seq_to_cap(seq))
                print(scores[i])

            print("Ground Truth")
            truth = data.return_groundtruth(img_name)
            for seq in truth:
                print(data.seq_to_cap(seq))

            img = data.return_img(img_name)
            plt.imshow(img)
            plt.show()

