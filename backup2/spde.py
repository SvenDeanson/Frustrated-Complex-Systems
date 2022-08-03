# -*- coding: utf-8 -*-

import numpy as np
import pulp as plp
import random
import matplotlib.pyplot as plt
import time
import networkx as nx
from tqdm import tqdm


from all_functions import *


def Calculate_Probabilites(size,prop):
    probabilites = []
    edges = []
    for i in range(10):
        avg=[]
        for k in tqdm(range(1000)):

            H = FG(size,prop)

            edges.append(len(H.copy().edges()))
            f = frustration_count(H.copy())
            avg.append(f)

        prob=[]
        for i in range(0,max(edges)):
            prob.append(np.count_nonzero(np.array(avg) == i))
        probabilites.append(prob)

    total = np.array(probabilites)
    final = []
    for i in range(0,len(total[0])):
        final.append((np.mean(total[:,i])/1000,np.std(total[:,i])/1000))

    return np.array(final)



G = Create_Random_Data(5,4,k=0.5)


Properties(G)




