# Cosine Similarity
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import networkx as nx
import Levenshtein
import random
import itertools
from tensorflow.keras.utils import Progbar

def processData(rawContents):
    return rawContents.split(' ')

def dfCoSim(tfs,fileNames):
    numFiles = len(fileNames)
    progbar = Progbar(numFiles)
    a = []
    for i in range(numFiles):
        a_temp = []
        for j in range(numFiles):
            matrixValue = cosine_similarity(tfs[i], tfs[j])
            numValue = round(float(matrixValue[0][0]),3)
            a_temp.append(numValue)
        a.append(a_temp)
        progbar.update(i+1)
    print("Finished.")
    return pd.DataFrame(a,columns=fileNames,index=fileNames)

def remove_similar(data_relevant, co_sim, threshold):
    dat = deepcopy(data_relevant)
    cosim = deepcopy(co_sim)
    
    def compare(j,threshold=threshold,cosim=cosim,dat=dat):
        matches = pd.Series(range(len(dat))).apply(lambda i: cosim.iloc[i,j]>=threshold)
        # get positive matches
        matches = matches[matches].index.tolist()
        # convert to list of tuples
        return [*zip(iter(matches[:-1]), iter(matches[1:]))]
    
    # create graph objects
    nodes = [i for i in range(len(dat))]
    edges = [*itertools.chain(*pd.Series(range(len(dat))).apply(compare))]

    # create graphs
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # get connected component indexes
    grouped_indexes = [*nx.connected_components(G)]

    # kalo ada berita yg sama, ambil berita yg paling duluan diupload
    filtered_indexes = np.sort([np.sort(list(_))[::-1][0] for _ in grouped_indexes]) 

    data_relevant_filtered = dat.iloc[filtered_indexes]
    return grouped_indexes,filtered_indexes,data_relevant_filtered