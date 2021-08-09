import pandas as pd
from copy import deepcopy
from nltk.tokenize import word_tokenize, sent_tokenize
import random
import numpy as np
from withpos.preprocessing import getCasing, convLabels
from tensorflow.keras.utils import Progbar

def toLabeledNER(df,col):
    df_ = deepcopy(df)
    se_li = [] ; wo_li = [] ; tag_li  = []; art_li = []
    sent_cs = 0
    for a in df_.index:
        se_li.append(""); wo_li.append("----------DOCSTART----------"); tag_li.append(""); art_li.append(str(a))
        tag_count = tag_cs = 0
        art = df_.loc[a,col].split()
        word = ' '.join(art[0::2]).strip(); tag = ' '.join(art[1::2]).strip()
        assert(len(word.split())==len(tag.split()))
        for i,x in enumerate(sent_tokenize(word)):
            for j,y in enumerate(x.split()):
                tag_count = tag_cs+j
                art_li.append(str(a)); se_li.append(str(i+sent_cs)) ; wo_li.append(y) ; tag_li.append(tag.split()[tag_count])
            tag_cs = tag_count+1
        sent_cs = i+sent_cs+1
    assert(len(se_li)==len(wo_li)==len(tag_li)) 
    df = pd.DataFrame({'article': art_li,
                       'sentence':se_li,
                       'word': wo_li,
                       'pos':tag_li})
    return df

def readLabeled(FILE_DIR):
    col_names = pd.read_csv(FILE_DIR,nrows=0,index_col=0).columns
    types_dict = {}
    types_dict.update({col: str for col in col_names if col not in types_dict})
    data = pd.read_csv(FILE_DIR,dtype=types_dict,index_col=0)
    return data

def toArray(df):
    array = []
    for s in [str(j) for j in sorted([int(i) for i in np.unique(df.sentence)])]:
        array_sent = []
        sent = df[df.sentence==s]
        for t in range(len(sent)):
            array_sent.append([sent.iloc[t,2],sent.iloc[t,4],sent.iloc[t,3]]) # [tok, pos, ner]
        array.append(array_sent)
    return array

def addCharInformation(Sentences):
    for i, sentence in enumerate(Sentences):
        for j, data in enumerate(sentence):
            chars = [c for c in data[0]]
            Sentences[i][j] = [data[0], chars, data[1], data[2]]
    return Sentences

def createMatrices(sentences, word2Idx, label2Idx, case2Idx, char2Idx, pos2Idx): # berhubungan juga dengan getCasing
    unknownIdx = word2Idx['UNKNOWN_TOKEN']
    paddingIdx = word2Idx['PADDING_TOKEN']

    dataset = []

    wordCount = 0
    unknownWordCount = 0

    # iterasi setiap kalimat
    for sentence in sentences:
        wordIndices = []
        caseIndices = []
        charIndices = []
        labelIndices = []
        posIndices = []
        
        #iterasi word, char, label dalam kalimat
        for word, char, label, pos in sentence:
            wordCount += 1
            if word in word2Idx:
                wordIdx = word2Idx[word]
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]
            else:
                wordIdx = unknownIdx
                unknownWordCount += 1
            charIdx = []
            for x in char:
                charIdx.append(char2Idx[x])
            # Get the label and map to int
            wordIndices.append(wordIdx)
            caseIndices.append(getCasing(word, case2Idx))
            charIndices.append(charIdx)
            labelIndices.append(label2Idx[label])
            posIndices.append(pos2Idx[pos])

        dataset.append([wordIndices, caseIndices, charIndices, labelIndices, posIndices])

    return dataset

# return batches ordered by words in sentence
def createBatches(data): #ngurutin sentence yang panjang word nya sama.
    l = []
    for i in data:
        l.append(len(i[0]))
    l = list(np.sort(list(set(l))))
    batches = []
    batch_len = []
    tokidx = []
    z = 0
    for i in l:
        idx = 0
        for batch in data:
            if len(batch[0]) == i:
                batches.append(batch)
                tokidx.append(idx)
                z += 1
            idx += 1
        batch_len.append(z)
    return tokidx,batches,batch_len

def tag_dataset(self, dataset, model):
    correctLabels = []
    predLabels = []
    progbar = Progbar(len(dataset))
    #print("Tagging token..")
    for i, data in enumerate(dataset):
        tokens, casing, char, labels, pos = data
        tokens = np.asarray([tokens])
        casing = np.asarray([casing])
        char = np.asarray([char])
        pos = np.asarray([pos])
        pred = model.predict([tokens, casing, char, pos], verbose=False)[0]#disinilah dia memprediksi modelnya
        progbar.update(i+1)
        pred = pred.argmax(axis=-1)  # Predict the classes
        correctLabels.append(labels)
        predLabels.append(pred)
    #print("Done.")
    return correctLabels, predLabels