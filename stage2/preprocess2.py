
# coding: utf-8

import numpy as np
import pandas as pd
import random
import jieba

test_path = '/search/work/input/data' # test set

test_set = pd.read_json(test_path, orient='records', encoding='utf-8', lines=True)
print ('test data loaded')

# cut words, remove punctuation, lower case
def preprocess(text, aug=False):
    jieba.suggest_freq(('不', '会'), tune=True)
    jieba.suggest_freq(('不', '能'), tune=True)
    jieba.suggest_freq(('不', '行'), tune=True)
    jieba.suggest_freq(('不', '好'), tune=True)
    jieba.suggest_freq(('不', '要'), tune=True)
    jieba.suggest_freq(('不', '是'), tune=True)
    jieba.suggest_freq(('不'), tune=True)
    jieba.suggest_freq('无法确定', tune=True)
    sent = jieba.lcut(text, HMM=False)
    for i in range(len(sent)):
        if sent[i] in "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）《》(•ؔʶ ˡ̲̮ ؔʶ)✧“”：【】]+":
            sent[i] = ' '
        elif aug and random.random()<0.1: # data augmentation
            sent[i] = ' '
        else:
            sent[i].lower()
    sent = ' '.join(sent)
    return sent

# concatenate query and alternatives
def query_alt(query, alternatives, a):
    '''
    query: line['query'] from original dataframe
    alternatives: line['alternatives'] from original dataframe
    a: current option in alternatives to be merged with query
    
    return: query and current option a concatenated (preprocessed)
    '''
    
    query = query.strip()
    if query[-1] == "吗" or query[-1] == "么" or query[-1] == "嘛" or query[-1] == "不": 
        query = query[:-1]
        match = None
        o = alternatives.split('|')
        o = [m.strip() for m in o]
        if '无法确认' in o:
            o.remove('无法确认')
        if '无法确定' in o:
            o.remove('无法确定') 
        if o[0] in o[1]:
            long = o[1]
            short = o[0]
        else:
            long = o[0]
            short = o[1]
        if long in query:
            match = long
        else:
            if short in query:
                match = short
            elif (short == '能') and ('可以' in query):
                match = '可以'
            elif (short == '可以') and ('能' in query):
                match = '能'
            elif (short == '可以') and ('会' in query):
                match = '会'
            elif (short == '会') and ('可以' in query):
                match = '可以'
            elif (short == '会') and ('能' in query):
                match = '能'
            elif (short == '能') and ('会' in query):
                match = '会'

        if match:
            query = query.replace(match, a)
        else:
            query = a + query
            
        merged = preprocess(query, alternatives)
        return merged
            
    else: # 问题里正反两个词都要替换
        match = alternatives.split('|')
        match = [m.strip() for m in match]
        if '无法确认' in match:
            match.remove('无法确认')
        if '无法确定' in match:
            match.remove('无法确定') 
        if match[0] in query and match[1] in query: # 两个词都出现了
            if match[0] + match[1] in query: # 有没有，会不会
                query = query.replace(match[0] + match[1], a)
            elif match[1] + match[0] in query:
                query = query.replace(match[1] + match[0], a)
            else: # A好还是B好
                if a == match[0]:
                    query = query.replace(match[1], ' ')
                elif a == match[1]:
                    query = query.replace(match[0], ' ')
                else: # 无法确定
                    query = query.replace(match[0], ' ')
                    query = query.replace(match[1], a)
        else: # 两个词没完整出现
            if '能否' in query:
                query = query.replace('能否', a)
            elif '是否' in query:
                query = query.replace('是否', a)
            elif '可否' in query:
                query = query.replace('可否', a)
            
        merged = preprocess(query, alternatives)
        return merged
    
# write original data into tsv file, each sample split into three pairs
print ('text preprocessing......')
with open('/search/work/output/test.tsv', 'w', encoding='utf-8') as fw:
    fw.write('id' + '\t' + 'passage' + '\t' + 'query' + '\t'+ 'option'+ '\n')
    for i in range(test_set.shape[0]):
        line = test_set.iloc[i]
        p = preprocess(line['passage'], line['alternatives'])
        for a in line['alternatives'].split('|'):
            a = a.strip()
            m = query_alt(query=line['query'], alternatives=line['alternatives'], a=a)
            fw.write(str(line['query_id'])+ '\t'+ p+ '\t'+ m+ '\t'+ a+ '\n')
print ('text preprocessed and archived')

# Add two manual feature: Exact match and option match
maxlen_p = 150
maxlen_q = 15

test_pair = '/search/work/output/test.tsv' # test set

test = pd.read_csv(test_pair, sep='\t', header=0)

print ('manual feature calculating......')
pl = []
ql = []
for i in range(test.shape[0]):
    line = test.iloc[i]
    q_words = line['query'].split()
    p_words = line['passage'].split()
    option = line['option']
    
    if len(p_words) > maxlen_p: # truncate pre
        lt = len(p_words) - maxlen_p
        p_words = p_words[lt:]
    if len(q_words) > maxlen_q: # truncate post
        q_words = q_words[:maxlen_q]

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
        
    while len(pfea) < maxlen_p: # pad with 0 pre
        pfea.insert(0, [0] * 2)
    pl.append(pfea)
    while len(qfea) < maxlen_q: # pad with 0 post
        qfea.append([0] * 2)
    ql.append(qfea)

pl = np.asarray(pl)
ql = np.asarray(ql)
np.save('/search/work/output/test_fea_p', pl)
np.save('/search/work/output/test_fea_q', ql)
print ('manual feature archived')

print ('2nd manual feature calculating......')
pl = []
ql = []
for i in range(test.shape[0]):
    line = test.iloc[i]
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
np.save('/search/work/output/test_fea_p2', pl)
np.save('/search/work/output/test_fea_q2', ql)
print ('2nd manual feature archived')