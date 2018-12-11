
# coding: utf-8

from keras.models import Model
from keras.layers.core import Dense, Dropout, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate, Concatenate, multiply, subtract, add, dot
from keras.layers import  Input, SpatialDropout1D, Bidirectional
from keras.layers import CuDNNLSTM, Conv1D
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

def match (x):
    p = x[0] # [t, 2d]
    q = x[1] # [t, 2d]
    prod = multiply([p, q]) # [t, 2d]
    sub = subtract([p, q]) # [t, 2d]
    conc = concatenate([p, q, prod, sub]) # [t, 8d]
    m = Conv1D(filters=K.int_shape(p)[-1], kernel_size=1, activation='tanh')(conc) # [t, 2d]
    return m
def gate (x):
    p = x[0] # [t, 2d]
    q = x[1] # [t, 2d]
    prod = multiply([p, q]) # [t, 2d]
    sub = subtract([p, q]) # [t, 2d]
    conc = concatenate([p, q, prod, sub]) # [t, 8d]
    g = Conv1D(filters=K.int_shape(p)[-1], kernel_size=1, activation='sigmoid')(conc) # [t, 2d]
    return g
def fuse(x):
    p = x[0] # [t, 2d]
    q = x[1] # [t, 2d]
    m = match(x) # [t, 2d]
    g = gate(x) # [t, 2d]
    f = add([multiply([g, m]), multiply([subtract([K.ones_like(g), g]), p])]) # [t, 2d]
    return f
def hafn (x):
    p = x[0] # [t, 2d]
    q = x[1] # [j, 2d]
    
    # co-attention & fusion
    s = K.batch_dot(p, K.permute_dimensions(q, (0,2,1)), axes=[2,1]) # [t, j]
    p2q = K.batch_dot(K.softmax(s, axis=-1), q, axes=[2,1]) # [t, 2d]
    q2p = K.batch_dot(K.permute_dimensions(K.softmax(s, axis=-2), (0,2,1)), p, axes=[2,1]) # [j, 2d]
    pf = fuse([p, p2q]) # [t, 2d]
    qf = fuse([q, q2p]) # [j, 2d]
    
    # self-attention & fusion for passage
    d = Bidirectional(CuDNNLSTM(num_rnn_units, return_sequences=True))(pf) # [t, 2d]
    sd = K.batch_dot(d, K.permute_dimensions(d, (0,2,1)), axes=[2,1]) # [t, t]
    d2d = K.batch_dot(K.softmax(sd, axis=-1), d, axes=[2,1]) # [t, 2d]
    df = fuse([d, d2d]) # [t, 2d]
    df = Bidirectional(CuDNNLSTM(num_rnn_units, return_sequences=True))(df) # [t, 2d]
    
    # self-align the query
    qf = Bidirectional(CuDNNLSTM(num_rnn_units, return_sequences=True))(qf) # [j, 2d]
    wq = Conv1D(filters=1, kernel_size=1, activation='softmax', use_bias=False)(qf) # [j, 1]
    q = K.batch_dot(qf, K.squeeze(wq, -1), axes=[1,1]) # [2d]
    
    # self defined q2p attention
    a = K.softmax(K.batch_dot(df, q, axes=[2,1]), axis=-1) # [t]
    r = K.batch_dot(df, a, axes=[1,1]) # [2d]
    y = concatenate([r, q]) # [4d]
    return y

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
    h = Bidirectional(CuDNNLSTM(num_rnn_units, return_sequences=True))(pf) # [t, 2d]
    u = Bidirectional(CuDNNLSTM(num_rnn_units, return_sequences=True))(qe) # [j,2d]

    # Hierachical attention fusion network
    y = Lambda(hafn)([h, u]) # [4d]
    
    um = Bidirectional(CuDNNLSTM(num_rnn_units, return_sequences=False))(u) # [2d]

    # Output layer
    conc = Concatenate()([y, um])
    x = BatchNormalization()(conc)
    x = Dense(num_hidden_units, activation='relu')(x)
    x = Dropout(drop_prob)(x)
    x = BatchNormalization()(x)

    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[p, q, p_fea, q_fea], outputs=x)
    return model
