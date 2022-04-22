#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 10:43:51 2022

@author: sven
"""


from all_functions import *

from collections import Counter

import collections


def Graph_Delta(N,choice):

    g = Create_Random_Data(N,choice)
    
    return g,calculate_delta(g)


def Create_SET_DELTA_Data(size,choice,target):

    Gs = []
    mydict = collections.defaultdict(list)
    for index in tqdm(range(10000)):
        
         g,d = Graph_Delta(N,choice)

         Gs.append(g)
         mydict[d].append(index)
    
# =============================================================================
#     
#     for i in tqdm(range(10000)):
#         
#         g = Create_Random_Data(size,choice)
#         Gs.append(g)
#         #if frustration_count(g) != calculate_delta(g):
#             #Gs.append(g)
# 
#     #dic = {}
#     mydict = collections.defaultdict(list)
#     for index,gs in tqdm(enumerate(Gs)):
#         mydict[str(calculate_delta(gs))].append(index)
#         
# =============================================================================
  
    #AVG = np.array(avg)
    
    #x = input('Enter your choice of delta:')
    #x = random.choice(AVG)

    indexs  = mydict[target]
    accessed_mapping = map(Gs.__getitem__, indexs)
    selected_graphs = list(accessed_mapping)
       
    return selected_graphs


N = 10
choice = 4
target = 0

g0 = Create_Random_Data(N,choice)
edge_length = len(g0.edges())

print("\nL=",edge_length,"\n")

#arrays  = []
for i in range(3):
    #print(i)
    Gs = Create_SET_DELTA_Data(N,choice,target)
    
    
    
    directory = "batch"+str(i)
  
    # Parent Directory path
    parent_dir = "/home/sven/Desktop/graph_data2/"
      
    # Path
    path = os.path.join(parent_dir, directory)
      

    #os.mkdir(path)

    os.mkdir(path)
    
    print("\nDirectory '% s' created" % directory,"\n")   
    
    for index,g in enumerate(Gs): 
    
        nx.write_gpickle(g, "/home/sven/Desktop/graph_data2/"+str(directory)+"/test"+str(index)+".gpickle")






#print(arrays)
    
#Calculate_Probabilites(arrays,edge_length)

# =============================================================================
# 
# fig, ax = plt.subplots(figsize=(8,8))
# 
# ax.plot([i for i in range(len(final))],[i for i,j in final])
# 
# 
# y = np.array([i for i,j in final])
# y_err = np.array([j for i,j in final])
# x = [i for i in range(len(final))]
# 
# ax.fill_between(x, y - y_err, y + y_err,facecolor="blue",alpha=0.2)
# 
# =============================================================================
