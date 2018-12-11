
# coding: utf-8

import numpy as np
import pandas as pd
# ## Data file path

maxlen_p = 150
maxlen_q = 15


# In[3]:


train_path = '/search/work/train.tsv' # train set
valid_path = '/search/work/valid.tsv' # validation set


# In[4]:

print ('train feature calculating.......')
train = pd.read_csv(train_path, sep='\t', header=0)
valid = pd.read_csv(valid_path, sep='\t', header=0)

pl = []
ql = []
for i in range(train.shape[0]):
    line = train.iloc[i]
    q_words = line['query'].split()
    p_words = line['passage'].split()
    option = line['option']
    
    if len(q_words) > maxlen_q: # truncate pre
        lt = len(q_words) - maxlen_q
        q_words = q_words[lt:]
    if len(p_words) > maxlen_p: # truncate post
        p_words = p_words[:maxlen_p]

    pfea = []
    for w in p_words:
        # exact match
        if w in q_words:
            em = 1
        else:
            em = 0
        # option match
        if w == option:
            om = 1
        else:
            om = 0
        pfea.append([em, om])
        
    qfea = []
    for w in q_words:
        # exact match
        if w in p_words:
            em = 1
        else:
            em = 0
        # option match
        if w == option:
            om = 1
        else:
            om = 0
        qfea.append([em, om])
        
    while len(qfea) < maxlen_q: # pad with 0 pre
        qfea.insert(0, [0] * 2)
    ql.append(qfea)
    while len(pfea) < maxlen_p: # pad with 0 post
        pfea.append([0] * 2)
    pl.append(pfea)

pl = np.asarray(pl)
ql = np.asarray(ql)
np.save('/search/work/train_fea_p2', pl)
np.save('/search/work/train_fea_q2', ql)
print (np.shape(pl), np.shape(ql))


# In[6]:
print ('valid feature calculating.......')

pl = []
ql = []
for i in range(valid.shape[0]):
    line = valid.iloc[i]
    q_words = line['query'].split()
    p_words = line['passage'].split()
    option = line['option']
    
    if len(q_words) > maxlen_q: # truncate pre
        lt = len(q_words) - maxlen_q
        q_words = q_words[lt:]
    if len(p_words) > maxlen_p: # truncate post
        p_words = p_words[:maxlen_p]

    pfea = []
    for w in p_words:
        # exact match
        if w in q_words:
            em = 1
        else:
            em = 0
        # option match
        if w == option:
            om = 1
        else:
            om = 0
        pfea.append([em, om])
        
    qfea = []
    for w in q_words:
        # exact match
        if w in p_words:
            em = 1
        else:
            em = 0
        # option match
        if w == option:
            om = 1
        else:
            om = 0
        qfea.append([em, om])
        
    while len(qfea) < maxlen_q: # pad with 0 pre
        qfea.insert(0, [0] * 2)
    ql.append(qfea)
    while len(pfea) < maxlen_p: # pad with 0 post
        pfea.append([0] * 2)
    pl.append(pfea)

pl = np.asarray(pl)
ql = np.asarray(ql)
np.save('/search/work/valid_fea_p2', pl)
np.save('/search/work/valid_fea_q2', ql)
print (np.shape(pl), np.shape(ql))
