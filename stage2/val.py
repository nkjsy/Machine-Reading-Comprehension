
# coding: utf-8

import numpy as np
import pandas as pd

import match0, BIDAF6

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import optimizers
from keras import backend as K

embed_size = 300 # how big is each word vector
max_features = 160000 # how many unique words to use (i.e num rows in embedding vector)
maxlen_p = 150 # max number of words in a context to use
maxlen_q = 15 # max number of words in a question to use
max_norm = 5.0
features = 2
batch_size = 256

train_path = '/search/work/train.tsv' # train set
valid_path = '/search/work/valid.tsv' # valid set
test_path = '/search/work/output/test.tsv' # test set
embed_file = '/search/work/sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5' # 预训练词向量
fasttext_file = '/search/work/cc.zh.300.vec' # 预训练词向量
train_feature_p_path = '/search/work/train_fea_p2.npy' # train passage word feature
valid_feature_p_path = '/search/work/valid_fea_p2.npy' # validation passage word feature
train_feature_q_path = '/search/work/train_fea_q2.npy' # train passage word feature
valid_feature_q_path = '/search/work/valid_fea_q2.npy' # validation passage word feature
test_feature_p_path = '/search/work/output/test_fea_p2.npy' # test passage word feature
test_feature_q_path = '/search/work/output/test_fea_q2.npy' # test query word feature

# Read file
train = pd.read_csv(train_path, sep='\t', header=0)
valid = pd.read_csv(valid_path, sep='\t', header=0)
test = pd.read_csv(test_path, sep='\t', header=0)
train_feature_p = np.load(train_feature_p_path)
train_feature_q = np.load(train_feature_q_path)
valid_feature_p = np.load(valid_feature_p_path)
valid_feature_q = np.load(valid_feature_q_path)
test_feature_p = np.load(test_feature_p_path)
test_feature_q = np.load(test_feature_q_path)

vl = valid.shape[0]
v = valid
valid = train.iloc[:vl]
train = pd.concat([v, train.iloc[vl:]], ignore_index=True)

v = valid_feature_p
valid_feature_p = train_feature_p[:vl,:,:]
train_feature_p = np.concatenate((v, train_feature_p[vl:,:,:]), axis=0)

v = valid_feature_q
valid_feature_q = train_feature_q[:vl,:,:]
train_feature_q = np.concatenate((v, train_feature_q[vl:,:,:]), axis=0)
print ('file loaded')

# Fit the tokenizer on train, valid and test set
tokenizer = Tokenizer(num_words=max_features, lower=True) 

tokenizer.fit_on_texts(pd.concat([train['passage'], train['query'], valid['passage'], valid['query'], test['passage'], test['query']], ignore_index=True))

tra_p = tokenizer.texts_to_sequences(train['passage'])
tra_q = tokenizer.texts_to_sequences(train['query'])
val_p = tokenizer.texts_to_sequences(valid['passage'])
val_q = tokenizer.texts_to_sequences(valid['query'])
te_p = tokenizer.texts_to_sequences(test['passage'])
te_q = tokenizer.texts_to_sequences(test['query'])

train_p = pad_sequences(tra_p, maxlen=maxlen_p, padding='post', truncating='post')
train_q = pad_sequences(tra_q, maxlen=maxlen_q)
valid_p = pad_sequences(val_p, maxlen=maxlen_p, padding='post', truncating='post')
valid_q = pad_sequences(val_q, maxlen=maxlen_q)
test_p = pad_sequences(te_p, maxlen=maxlen_p, padding='post', truncating='post')
test_q = pad_sequences(te_q, maxlen=maxlen_q)

train_l = train['label']
valid_l = valid['label']

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

def train1(model, path, epoch):
    adam = optimizers.Adam(lr=0.001, clipnorm=max_norm)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['binary_accuracy'])
    cp = ModelCheckpoint(filepath=path, monitor='val_binary_accuracy', save_best_only=True, save_weights_only=True)
    es = EarlyStopping(patience=0,  monitor='val_binary_accuracy')
    rp = ReduceLROnPlateau(patience = 1,  monitor='val_loss')
    hist = model.fit(
        [train_p, train_q, train_feature_p, train_feature_q], 
        train_l,
        batch_size = batch_size,
        epochs = epoch,
        shuffle = True,
        validation_data = ([valid_p, valid_q, valid_feature_p, valid_feature_q], valid_l), 
        callbacks=[rp, cp, es])
    
def train2(model, path):        
    # Train the model again
    adam = optimizers.Adam(lr=0.0005, clipnorm=max_norm)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['binary_accuracy'])
    cp = ModelCheckpoint(filepath=path, monitor='val_binary_accuracy', save_best_only=True, save_weights_only=True)
    hist = model.fit(
        [train_p, train_q, train_feature_p, train_feature_q], 
        train_l,
        batch_size = batch_size,
        epochs = 1,
        shuffle = True,
        validation_data = ([valid_p, valid_q, valid_feature_p, valid_feature_q], valid_l), 
        callbacks=[cp])

def predict(model, p):
    # predict the test data
    test_pred = model.predict([test_p, test_q, test_feature_p, test_feature_q], batch_size=batch_size*2)
    test_pred = np.squeeze(test_pred)
    
    # Write the array into csv file
    
    res = pd.DataFrame({'id':test['id'], 'option':test['option'], 'label':test_pred})
    res.to_csv(p, index=False, encoding='utf-8_sig')
'''
# match-lstm
K.clear_session()
model = match0.single_model(embedding_matrix, fasttext_matrix)
train1(model, '/search/work/output/my3.h5', 1)
# load best weights
model.load_weights('/search/work/output/my3.h5')
predict(model, '/search/work/output/test3_long.csv')
print ('match prediction saved')
'''
# BIDAF
K.clear_session()
model = BIDAF6.single_model(embedding_matrix, fasttext_matrix)
train1(model, '/search/work/output/my3.h5', 5)
# load best weights
model.load_weights('/search/work/output/my3.h5')
predict(model, '/search/work/output/test3_long.csv')
print ('BIDAF prediction saved')