import pandas as pd
import numpy as np
import random
from withpos.preprocessing import getCasing, convLabels
from tensorflow.keras.utils import Progbar


def toArray_new(df):
    array = []
    for s in [str(j) for j in sorted([int(i) for i in np.unique(df.sentence)])]:
        array_sent = []
        sent = df[df.sentence==s]
        for t in range(len(sent)):
            array_sent.append([sent.iloc[t,2],sent.iloc[t,3]]) # [tok, pos, ner]
        array.append(array_sent)
    return array

# return batches ordered by words in sentence
# New batches gada bedanya wei
def createBatches_new(data): #ngurutin sentence yang panjang word nya sama.
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


# returns matrix with 1 entry = list of 3 elements:
# word indices, case indices, character indices, label indices
def createMatrices_new(sentences, word2Idx, case2Idx, char2Idx, pos2Idx): # berhubungan juga dengan getCasing
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
        posIndices = []
        
        #iterasi word, char, pos dalam kalimat
        for word, char, pos in sentence:
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
            posIndices.append(pos2Idx[pos])

        dataset.append([wordIndices, caseIndices, charIndices, posIndices])

    return dataset

# returns data with character information in format
# [['EU', ['E', 'U'], 'B-ORG\n'], ...]
def addCharInformation_new(Sentences):
    for i, sentence in enumerate(Sentences):
        for j, data in enumerate(sentence):
            chars = [c for c in data[0]]
            #print(data[0])
            Sentences[i][j] = [data[0], chars, data[1]]
    return Sentences

def tag_dataset_new(dataset, model):
    predLabels = []
    progbar = Progbar(len(dataset))
    print("Tagging token..")
    for i, data in enumerate(dataset):
        tokens, casing, char, pos = data
        tokens = np.asarray([tokens])
        casing = np.asarray([casing])
        char = np.asarray([char])
        pos = np.asarray([pos])
        progbar.update(i+1)
        pred = model.predict([tokens, casing, char, pos], verbose=False)[0]#disinilah dia memprediksi modelnya
        pred = pred.argmax(axis=-1)  # Predict the classes
        predLabels.append(pred)
    print("Done.")
    return predLabels

def get_prediction_ner_new(data,prediction,batch_idx,idx2Label): 
    pred = convLabels(prediction, idx2Label)
    output = []
    for i,j in enumerate(batch_idx): # foreach sentence
        temp = []
        for x,y in enumerate(data[j]): # foreach word
            temp.append([y[0],pred[i][x]])  # [word, ner, pred_ner]
        temp.append(j)
        output.append(temp)
    output = sorted(output,key= lambda x: x[-1])
    output_ = [out[:-1] for out in output]
    return output_

def prediction_ner_to_df_new(sentences_, data):
    idx = data.index
    df = pd.DataFrame(columns=['word','pred_ner'], index = idx)
    row_count = 0
    for sent in sentences_:
        for tok in sent:
            df.loc[idx[row_count]] = tok
            row_count+=1
    return df

def tag_dataset_new(dataset, model):
    predLabels = []
    progbar = Progbar(len(dataset))
    print("Tagging token..")
    for i, data in enumerate(dataset):
        tokens, casing, char, pos= data
        tokens = np.asarray([tokens])
        casing = np.asarray([casing])
        char = np.asarray([char])
        pos = np.asarray([pos])
        progbar.update(i+1)
        pred = model.predict([tokens, casing, char, pos], verbose=False)[0]#disinilah dia memprediksi modelnya
        pred = pred.argmax(axis=-1)  # Predict the classes
        predLabels.append(pred)
    print("Done.")
    return predLabels