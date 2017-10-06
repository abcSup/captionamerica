from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Reshape, Lambda, RepeatVector, Permute
from keras.layers import Activation, GlobalAveragePooling2D, GlobalAveragePooling1D, Conv1D
from keras.layers import LSTM, Bidirectional, TimeDistributed
from keras.layers.merge import Concatenate, Add, Multiply
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import RNN
from keras.optimizers import RMSprop
from keras import backend as K
from keras import metrics

from lstm_vs import LSTMCell_VS, RNN_VS

def image_model():
    
    input = Input((299,299,3))
    base_model = InceptionV3(input_tensor=input, weights='imagenet', include_top=False)

    #x = base_model.output
    #x = GlobalAveragePooling2D()(x)
    #x = Dense(embedding_size)(x)
    #x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    #x = Dropout(0.2)(x)
    #x = Reshape((1, embedding_size))(x)

    for layer in base_model.layers:
        layer.trainable = False

    return base_model, input

def get_model():
    # first model
    # embedding = 512, seq = 36, vocab = 8045
    # dropout 0.8, lstm dropout 0.7
    #  lstm 1024

    emb_size = 512
    d_dim = 512
    vocab_size = 10000
    lstm_units = 512
    seq_len = 42
    dropout = True
    
    cnn, img_input = image_model()
    V = cnn.output

    # k regions, image length
    k_size = int(V.shape[1]) * int(V.shape[2])
    num_feats = int(V.shape[3])

    V = Reshape((k_size,num_feats))(V)

    # global image feature
    # k = regions
    Vg = GlobalAveragePooling1D(name='Vg')(V)
    Vg = Dense(emb_size, name='Vg.')(Vg)
    Vg = BatchNormalization(name='Vg..')(Vg)
    Vg = Activation('relu', name='Vg...')(Vg)
    if dropout:
        Vg = Dropout(0.5)(Vg)

    # Vi.shape (None, k, 512)
    Vi = Conv1D(d_dim, 1, padding='same', activation='relu')(V)
    if dropout:
        Vi = Dropout(0.5)(Vi)

    Vi_emb = Conv1D(emb_size, 1, padding='same', activation='relu')(Vi)

    x = RepeatVector(seq_len)(Vg)

    prev_words = Input((seq_len,))
    w_emb = Embedding(vocab_size, emb_size, 
                    input_length=seq_len, name='embedding')(prev_words)
    w_emb = Reshape((seq_len, emb_size))(w_emb)

    x = Concatenate(axis=2)([x, w_emb])

    lstmcell_vs = LSTMCell_VS(lstm_units)
    lstm = RNN_VS(lstmcell_vs, return_sequences=True)(x)
    # return_sequences true

    # (None, timestep, 512)
    h = Lambda(lambda x : x[:, :, :lstm_units], name='h')(lstm)
    s = Lambda(lambda x : x[:, :, lstm_units:], name='s')(lstm)

    # (None, k, timestep, 512)
    Vi = TimeDistributed(RepeatVector(seq_len), name='Vi.Repeat')(Vi)
    # (None, timestep, k, 512)
    Vi = Permute((2,1,3), name='Vi.Permute')(Vi)
    # (None, timestep, k, k)
    Wv_Vi = TimeDistributed(Dense(k_size), name='Wv_Vi')(Vi)
    
    # (None, timestep, k)
    Wg_h = TimeDistributed(Dense(k_size), name='Wg_h')(h)
    # (None, timestep, k, k)
    Wg_h_1 = TimeDistributed(RepeatVector(k_size), name='Wg_h_1')(Wg_h)

    z = Add(name='Wv_Vi_add_Wg_h_1')([Wv_Vi, Wg_h_1])
    z = TimeDistributed(Activation('tanh'), name='z_tanh')(z)

    # wTh, wth * tanh(z_Vi + z_h) 
    #z = TimeDistributed(Conv1D(1,1,padding='same'), name='wTh')(z)
    z = TimeDistributed(Dense(1), name='wTh')(z)
    # (None, timestep, k_size)
    z = Reshape((seq_len, k_size), name='wTh.Reshape')(z)

    # (None, timestep, k)
    Ws_s = TimeDistributed(Dense(k_size))(s)
    B = Add(name='Ws_s_add_Wg_h')([Ws_s, Wg_h])
    B = Dense(1)(B)

    a_hat = Concatenate(axis=2, name='z_B')([z, B])
    a_hat = TimeDistributed(Activation('softmax'), name='a_hat')(a_hat)

    # (None, timestep, 1)
    B = Lambda(lambda x: x[:, :, -1], name='B')(a_hat)
    att = Lambda(lambda x: x[:, :, :k_size], name='att')(a_hat)

    att = TimeDistributed(RepeatVector(d_dim), name='att.Repeat')(att)
    att = Permute((1,3,2), name='att.Permute')(att)

    # context vector
    # sum over k regions (att * Vi)
    ct = Multiply(name='a_multiply_v')([att, Vi])
    sum_pooling = Lambda(lambda x: K.sum(x, axis=-2),
                    output_shape=(d_dim,))
    # (None, timestep, 512)
    ct = TimeDistributed(sum_pooling, name='ct')(ct)

    B = RepeatVector(d_dim)(B)
    B = Permute((2,1))(B)
    B_inverse = Lambda(lambda x: 1-x, name='B_inverse')(B)

    B_s = Multiply(name='B_s')([B, s])
    B_ct = Multiply(name='B_ct')([B_inverse, ct])
    c_hat = Add(name='c_hat')([B_s, B_ct])

    c_hat_h = Add(name='c_hat_h')([c_hat, h])
    logits = TimeDistributed(Dense(vocab_size), name='logits')(c_hat_h)

    #logits = Dense(vocab_size)(lstm)
    #logits = BatchNormalization()(logits)
    softmax = Activation('softmax', name='softmax')(logits)

    model = Model(inputs=[img_input, prev_words], outputs=softmax)

    model.compile(optimizer=RMSprop(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def test_model():
    input = Input((32,512))
    lstmcell_vs = LSTMCell_VS(10)
    lstm = RNN_VS(lstmcell_vs)(input)

    h = Lambda(lambda x : x[:,:10])(lstm)
    s = Lambda(lambda x : x[:,10:])(lstm)

    h = Dense(1)(h)
    s = Dense(1)(s)
    
    model = Model(inputs=input, outputs=[h,s])

    model.compile(optimizer=RMSprop(),
                  loss='mse',
                  metrics=['accuracy'])

    return model

if __name__ == "__main__":
    model = get_model()
    model.summary()
