from keras.preprocessing.sequence import pad_sequences
import numpy as np

# define casing s.t. NN can use case information to learn patterns
def getCasing(word, caseLookup): # word disini yg langsung diambil dari sentence aslinya, jadi tau huruf besar kecilnya
    casing = 'other'

    numDigits = 0
    for char in word:
        if char.isdigit():
            numDigits += 1

    digitFraction = numDigits / float(len(word))

    if word.isdigit():  # Is a digit
        casing = 'numeric'
    elif digitFraction > 0.5:
        casing = 'mainly_numeric'
    elif word.islower():  # All lower case
        casing = 'allLower'
    elif word.isupper():  # All upper case
        casing = 'allUpper'
    elif word[0].isupper():  # is a title, initial char upper, then all lower
        casing = 'initialUpper'
    elif numDigits > 0:
        casing = 'contains_digit'

    return caseLookup[casing]

def iterate_minibatches(dataset, batch_len):
    start = 0
    for i in batch_len:
        tokens = []
        caseing = []
        char = []
        labels = []
        pos = []
        data = dataset[start:i]
        start = i
        for dt in data:
            t, c, ch, l, p = dt
            l = np.expand_dims(l, -1)
            tokens.append(t)
            caseing.append(c)
            char.append(ch)
            labels.append(l)
            pos.append(p)
        
        yield np.asarray(labels), np.asarray(tokens), np.asarray(caseing), np.asarray(char), np.asarray(pos)

# 0-pads all words
def padding(Sentences):
    maxlen = 52 # udah ditentukan ya cuy, ternyata panjang karakternya hehe
    for sentence in Sentences:
        char = sentence[2] #ambil character
        for x in char:
            maxlen = max(maxlen, len(x))
    for i, sentence in enumerate(Sentences):
        Sentences[i][2] = pad_sequences(Sentences[i][2], 52, padding='post')
    return Sentences

def convLabels(dataset,idx2Label):
    labels = []
    for sent in dataset:
        labels.append([idx2Label[word] for word in sent])
    return labels