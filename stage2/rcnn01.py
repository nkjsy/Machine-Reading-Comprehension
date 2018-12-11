
# coding: utf-8

import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Dropout, Reshape, Flatten, Lambda, Permute
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate, Concatenate, multiply, Dot, subtract, add, dot
from keras.layers import  GlobalMaxPooling1D, GlobalAveragePooling1D, Input, SpatialDropout1D, Bidirectional
from keras.layers import CuDNNLSTM, CuDNNGRU, LSTM, GRU, Conv1D
from keras import backend as K
from keras.engine.topology import Layer
from keras.preprocessing import sequence, text
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import optimizers

# hyper parameters
embed_size = 300 # how big is each word vector
max_features = 200000 # how many unique words to use (i.e num rows in embedding vector)
maxlen_p = 150 # max number of words in a context to use
maxlen_q = 15 # max number of words in a question to use
batch_size = 256
num_rnn_units = 128
num_hidden_units = 300
drop_prob = 0.4
max_norm = 5.0
features = 2
filter_sizes_p = [1,3,5]
filter_sizes_q = [1,3]
num_filters = 128

train_path = '/search/work/train.tsv' # train set
test_path = '/search/work/output/test.tsv' # test set
embed_file = '/search/work/sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5' # 预训练词向量
fasttext_file = '/search/work/cc.zh.300.vec' # 预训练词向量
test_feature_p_path = '/search/work/output/test_fea_p.npy' # test passage word feature
test_feature_q_path = '/search/work/output/test_fea_q.npy' # test query word feature


# Read file
train = pd.read_csv(train_path, sep='\t', header=0)
test = pd.read_csv(test_path, sep='\t', header=0)

test_feature_p = np.load(test_feature_p_path)
test_feature_q = np.load(test_feature_q_path)
print ('file loaded')

# Fit the tokenizer on train, valid and test set
tokenizer = Tokenizer(num_words=max_features, lower=True) 

tokenizer.fit_on_texts(pd.concat([train['passage'], train['query']], ignore_index=True))

te_p = tokenizer.texts_to_sequences(test['passage'])
te_q = tokenizer.texts_to_sequences(test['query'])

test_p = pad_sequences(te_p, maxlen=maxlen_p)
test_q = pad_sequences(te_q, maxlen=maxlen_q, padding='post', truncating='post')

# word embedding
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(embed_file, encoding='utf-8'))

all_embs = np.hstack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words+1, embed_size))
for word, i in word_index.items():
    if i > max_features: break
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

fasttext_index = dict(get_coefs(*o.strip().split()) for o in open(fasttext_file, encoding='utf-8'))
all_ft = np.hstack(fasttext_index.values())
ft_mean,ft_std = all_ft.mean(), all_ft.std()
fasttext_matrix = np.random.normal(ft_mean, ft_std, (nb_words+1, embed_size))
for word, i in word_index.items():
    if i > max_features: break
    fasttext_vector = fasttext_index.get(word)
    if fasttext_vector is not None: fasttext_matrix[i] = fasttext_vector
print ('embedding loaded')

# Build the model
K.clear_session()

def cos_sim (x):
    p = x[0] # [t, 2d]
    q = x[1] # [j, 2d]
    s = dot([p, K.permute_dimensions(q, (0,2,1))], axes=(2,1), normalize=True) # [t, j] cosine simlilarity
    max_sim = K.max(s, axis=-1, keepdims=True) # [t, 1]
    return max_sim

def single_model(trainable=False):
    p = Input(shape=(maxlen_p,))
    q = Input(shape=(maxlen_q,))
    p_fea = Input(shape=(maxlen_p, features)) # passage word feature (exact match + option match)
    q_fea = Input(shape=(maxlen_q, features)) # query word feature
    
    # Embedding layer
    embed = Embedding(nb_words+1, embed_size, weights=[embedding_matrix], trainable=trainable)
    ft = Embedding(nb_words+1, embed_size, weights=[fasttext_matrix], trainable=trainable)
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
    
    h = Bidirectional(CuDNNLSTM(num_rnn_units, return_sequences=True))(pf) # [t, 2d]
    u = Bidirectional(CuDNNLSTM(num_rnn_units, return_sequences=True))(qe) # [j,2d]
    
    # cnn for p
    convp_0 = Conv1D(num_filters, kernel_size=filter_sizes_p[0], padding = "valid", activation = 'relu')(h)
    convp_1 = Conv1D(num_filters, kernel_size=filter_sizes_p[1], padding = "valid", activation = 'relu')(h)
    convp_2 = Conv1D(num_filters, kernel_size=filter_sizes_p[2], padding = "valid", activation = 'relu')(h)

    maxpoolp_0 = GlobalMaxPooling1D()(convp_0)
    avgpoolp_0 = GlobalAveragePooling1D()(convp_0)
    maxpoolp_1 = GlobalMaxPooling1D()(convp_1)
    avgpoolp_1 = GlobalAveragePooling1D()(convp_1)
    maxpoolp_2 = GlobalMaxPooling1D()(convp_2)
    avgpoolp_2 = GlobalAveragePooling1D()(convp_2)
    zp = Concatenate()([maxpoolp_0, maxpoolp_1, maxpoolp_2, avgpoolp_0, avgpoolp_1, avgpoolp_2])

    # cnn for q
    convq_0 = Conv1D(num_filters, kernel_size=filter_sizes_q[0], padding = "valid", activation = 'relu')(u)
    convq_1 = Conv1D(num_filters, kernel_size=filter_sizes_q[1], padding = "valid", activation = 'relu')(u)

    maxpoolq_0 = GlobalMaxPooling1D()(convq_0)
    avgpoolq_0 = GlobalAveragePooling1D()(convq_0)
    maxpoolq_1 = GlobalMaxPooling1D()(convq_1)
    avgpoolq_1 = GlobalAveragePooling1D()(convq_1)
    zq = Concatenate()([maxpoolq_0, maxpoolq_1, avgpoolq_0, avgpoolq_1])  
    
    # Output layer
    x = Concatenate()([zp,zq])
    x = BatchNormalization()(x)
    x = Dense(num_hidden_units, activation='relu')(x)
    x = Dropout(drop_prob)(x)
    x = BatchNormalization()(x)
    #x = Dense(num_hidden_units, activation='relu')(x)
    #x = Dropout(drop_prob)(x)
    #x = BatchNormalization()(x)

    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[p, q, p_fea, q_fea], outputs=x)
    #print (model.summary())
    return model

model = single_model()

# load best weights
model.load_weights('/search/work/my5.h5')
print ('model loaded')

# predict the test data
test_pred = model.predict([test_p, test_q, test_feature_p, test_feature_q], batch_size=batch_size*4)
test_pred = np.squeeze(test_pred)

# Write the array into csv file

res = pd.DataFrame({'id':test['id'], 'passage':test['passage'], 'query':test['query'], 'option':test['option'], 'label':test_pred})
res.to_csv('/search/work/output/test5_long.csv', index=False, encoding='utf-8_sig')
print ('prediction saved')



