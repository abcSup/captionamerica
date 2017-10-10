from os.path import join
import numpy as np
import h5py

from keras.callbacks import ModelCheckpoint, TensorBoard

from configuration import Config
from model import get_model
from data_generator import DataGenerator
from data_generator import CustomSequence

nb_epoch = 200
#data_dir = './data'
#h5file = join(data_dir, 'train_data.h5') 
#data = h5py.File(h5file, 'r')
#seq_generator = CustomSequence(data, batch_size)

config = Config()
model = get_model(config)
data = DataGenerator(config)

checkpoint = ModelCheckpoint('./checkpoint/train_weight-{epoch:03d}-{loss:0.3f}.hdf5',
				monitor='loss',
				verbose=1,
				save_best_only=True,
				mode='min')
tensorboard = TensorBoard(log_dir='./logs',
				histogram_freq=0,
				write_graph=True,
				write_images=False)

#model.load_weights('./checkpoint/train_weight-009.hdf5')
model.fit_generator(data.generator(), config.num_batch,
                    epochs=nb_epoch,
                    initial_epoch=0,
                    callbacks=[tensorboard, checkpoint],
                    workers=1,
                    use_multiprocessing=False)
