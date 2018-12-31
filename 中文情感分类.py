#-*- coding:utf-8 -*-

import numpy as np
import sys,os

from keras.layers.core import Activation,Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing import text
from sklearn.model_selection import train_test_split
import nltk
import collections
import jieba






def load_text_vocabu(dataset,vocabu='/tmp/vocab.txt'):
    '''
    分析文本集合的分词词条，数据集行数，最大词条的长度，总的词条数
    :param dataset:
    :param vocabu:
    :return:
    '''
    num_rec = 0
    max_len = 0
    word_freq = collections.Counter()  # 初始化词频统计器
    voc=open(vocabu,'wt')
    label=[]
    data_X=[]
    with open(dataset,'rt',encoding='utf-8') as f:
        for line in f:
            seq=[]
            tt_line=line.strip('\n')
            dd=int(tt_line.split('\t')[0])
            if dd >0:
                dd=1
            label.append(dd)
            tmp_line=tt_line.split('\t')[1].replace(' ','')
            words=jieba.cut(tmp_line,cut_all=True)
            i=0
            for word in words:
                if word != '':
                    seq.append(word)
                    i+=1
                    word_freq[word]+=1
                    voc.write(word+'\n')
            if i >max_len:
                max_len=i
            num_rec+=1
            data_X.append(seq)
    voc.close()
    label=np.asarray(label,np.int8)

    return num_rec,max_len,word_freq,data_X,label

def embeded(word_freq,dataset,):

    #word2index = {x[0]: i + 2 for i, x in enumerate(word_freq.most_common(MAX_FEATURES))}
    # word2index["PAD"] = 0
    # word2index["UNK"] = 1
    word2index={"PAD":0,"UNK":1}
    tmp=word_freq.most_common(MAX_FEATURES)
    for i,x in enumerate(tmp):
        word2index[x[0]]=i+2

    index2word = {v: k for k, v in word2index.items()}

    X=[]
    with open(dataset,'rt') as f:
        for line in f:
            seq=[]
            tt_line = line.strip('\n')
            tmp_line = tt_line.split('\t')[1].replace(' ', '')
            words = jieba.cut(tmp_line, cut_all=True)
            for word in words:
                if word in word2index:
                    seq.append(word2index[word])
                else:
                    seq.append(word2index['UNK'])
            X.append(seq)

    X=sequence.pad_sequences(X,maxlen=MAX_SENTENCE_LENGTH)
    return X

def aderic_lstm():
    model = Sequential()
    model.add(Embedding(vocab_size, EMBEDDING_SIZE, input_length=MAX_SENTENCE_LENGTH))
    model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    return model

if __name__=='__main__':


    train_data_file=r'./corpus.train'
    train_vocab_f=r'./vocabu_train.txt'

    test_data_file=r'./corpus.test'
    test_vocab_f=r'./vocabu_test.txt'

    model_save_file_name=r'./lstm_chinese_comment_emotion_classcify'


    num_rec,max_len,word_freq,data_X,label=load_text_vocabu(train_data_file,train_vocab_f)
    print('train_data.txt, have ',num_rec,'条记录，单个记录最大词长度是',max_len,',总共',len(word_freq),'个词组！')



    num_rec_val,max_len_val,word_freq_val,data_X_val,label_val=load_text_vocabu(test_data_file,test_vocab_f)
    print('test_data.txt, have ',num_rec_val,'条记录，单个记录最大词长度是',max_len_val)


    ####feature param#########
    MAX_FEATURES=len(word_freq)-(len(word_freq)%100)
    MAX_SENTENCE_LENGTH=max_len-(max_len%100)
    vocab_size = min(MAX_FEATURES, len(word_freq)) + 2
    ####net param###########
    EMBEDDING_SIZE = 256
    HIDDEN_LAYER_SIZE = 64
    BATCH_SIZE = 64
    NUM_EPOCHS = 5
    ####param#########



    X=embeded(word_freq,train_data_file)

    X_val = embeded(word_freq_val, test_data_file)

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, label, test_size=0.2, random_state=42)

    model=aderic_lstm()
    model.fit(Xtrain,ytrain,batch_size=BATCH_SIZE,epochs=NUM_EPOCHS) #,validation_data=(X_val,label_val))
    model.save(model_save_file_name, )




    # from keras.models import load_model
    # model=load_model(model_save_file_name,)



    score, acc = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)
    print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))


