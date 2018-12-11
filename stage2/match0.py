
# coding: utf-8

from keras.models import Model
from keras.layers.core import Dense, Dropout, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate, Concatenate, dot
from keras.layers import  GlobalMaxPooling1D, GlobalAveragePooling1D, Input, SpatialDropout1D, Bidirectional
from keras.layers import CuDNNGRU
from keras import backend as K

# hyper parameters
embed_size = 300 # how big is each word vector
max_features = 160000 # how many unique words to use (i.e num rows in embedding vector)
maxlen_p = 150 # max number of words in a context to use
maxlen_q = 15 # max number of words in a question to use
num_rnn_units = 64
num_hidden_units = 300
drop_prob = 0.4
features = 2

# match-LSTM layer
def match_lstm (x):
    p = x[0] # [t, 2d]
    q = x[1] # [j, 2d]
    s = K.batch_dot(p, K.permute_dimensions(q, (0,2,1)), axes=[2,1]) # [t, j]
    p2q = K.batch_dot(K.softmax(s, axis=-1), q, axes=[2,1]) # [t, 2d]
    z = concatenate([p, p2q]) # [t, 4d]
    r = Bidirectional(CuDNNGRU(num_rnn_units, return_sequences=True))(z) # [t, 2d]
    return r

def cos_sim (x):
    p = x[0] # [t, 2d]
    q = x[1] # [j, 2d]
    s = dot([p, K.permute_dimensions(q, (0,2,1))], axes=(2,1), normalize=True) # [t, j] cosine simlilarity
    max_sim = K.max(s, axis=-1, keepdims=True) # [t, 1]
    return max_sim

def single_model(emb, fast, trainable=False):
    p = Input(shape=(maxlen_p,))
    q = Input(shape=(maxlen_q,))
    p_fea = Input(shape=(maxlen_p, features)) # passage word feature (exact match + option match)
    q_fea = Input(shape=(maxlen_q, features)) # query word feature
    
    # Embedding layer
    embed = Embedding(max_features+1, embed_size, weights=[emb], trainable=trainable)
    ft = Embedding(max_features+1, embed_size, weights=[fast], trainable=trainable)
    pem = embed(p) # word embedding
    pft = ft(p)
    pe = Concatenate()([pem, pft])
    qem = embed(q)
    qft = ft(q)
    qe = Concatenate()([qem, qft])
    
    p_cos_e = Lambda(cos_sim)([pem, qem])
    p_cos_f = Lambda(cos_sim)([pft, qft])
    q_cos_e = Lambda(cos_sim)([qem, pem])
    q_cos_f = Lambda(cos_sim)([qft, pft])
    pe = SpatialDropout1D(0.2)(pe)
    qe = SpatialDropout1D(0.2)(qe)
    pf = Concatenate()([pe, p_fea, p_cos_e, p_cos_f]) # passage feature vec = word embedding + (exact match + option match + cos sim)
    qe = Concatenate()([qe, q_fea, q_cos_e, q_cos_f]) # query feature vec = word embedding + (exact match + option match + cos sim)
    
    # Contextual embedding layer
    h = Bidirectional(CuDNNGRU(num_rnn_units, return_sequences=True))(pf) # [t, 2d]
    u = Bidirectional(CuDNNGRU(num_rnn_units, return_sequences=True))(qe) # [j,2d]

    # match-lstm layer
    r = Lambda(match_lstm)([h, u]) # [t, 2d]
    
    # Modelling layer
    m, hf, hb = Bidirectional(CuDNNGRU(num_rnn_units, return_sequences=True, return_state=True))(r) # [t, 2d], d, d, d, d
    um, uhf, uhb = Bidirectional(CuDNNGRU(num_rnn_units, return_sequences=True, return_state=True))(u) # [j,2d], d, d, d, d

    # Output layer
    conc = Concatenate()([r, m]) # [t, 4d]
    gmp = GlobalMaxPooling1D()(conc) # [4d]
    gap = GlobalAveragePooling1D()(conc) # [4d]
    z1 = Concatenate()([gmp, gap, hf, hb]) # [10d]
    
    ugmp = GlobalMaxPooling1D()(um) # [2d]
    ugap = GlobalAveragePooling1D()(um) # [2d]
    z2 = Concatenate()([ugmp, ugap, uhf, uhb]) # [6d]

    x = Concatenate()([z1, z2])
    x = BatchNormalization()(x)
    x = Dense(num_hidden_units, activation='relu')(x)
    x = Dropout(drop_prob)(x)
    x = BatchNormalization()(x)

    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[p, q, p_fea, q_fea], outputs=x)
    return model
