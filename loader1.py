#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 11:11:43 2022

@author: sven
"""

from all_functions import *
import os

arrays = []

basepath = '/home/sven/Desktop/graph_data/'
for folder in os.listdir(basepath):
    print(folder)
    
    graphs_array = []
    
    for filename in os.listdir(os.path.join(basepath, folder)):
        #print(filename)
        g = nx.read_gpickle(os.path.join(basepath,folder,filename))
        
        graphs_array.append(g)
        
    arrays.append(graphs_array)
        
# =============================================================================
# def Calculate_Probabilites0(arrays,edge_length):
#     
#     probabilites = []
#     for Gs in arrays:
#         avg=[]
#         for g in Gs:
#             
#             f = frustration_count(g.copy())
#             avg.append(f)
# 
#         prob=[]
#         for i in range(0,edge_length):
#             prob.append(np.count_nonzero(np.array(avg) == i))
#         probabilites.append(prob)
#         
#     total = np.array(probabilites)
#     final = []
#     for i in range(0,len(total[0])):
#         final.append((np.mean(total[:,i])/len(Gs),np.std(total[:,i])/len(Gs)))
# 
#     return np.array(final)
# 
# =============================================================================

def Calculate_Probabilites(arrays,edge_length):
    
    probabilites = []
    for Gs in arrays:
        avg=[]
        
        for index,g in enumerate(Gs):
        
            f = frustration_count(g.copy())
            avg.append(f)

        prob=[]
        for i in range(0,edge_length):
            prob.append(np.count_nonzero(np.array(avg) == i)/len(Gs))
        probabilites.append(prob)
        
        #print(probabilites)
        
        
    total = np.array(probabilites)
    final = []
    for i in range(0,len(total[0])):
        final.append((np.mean(total[:,i]),np.std(total[:,i])))

    return np.array(final)




edge_length = len(graphs_array[0].edges())
print("L = ",edge_length)




final = Calculate_Probabilites(arrays,edge_length)
#print(final)


filepath = "/home/sven/Desktop/l56d0.txt"
p_true = Load_Analytic_Solution(filepath)

print(p_true)
print(len(p_true))


# =============================================================================
# 
fig, ax = plt.subplots(figsize=(8,8))
# 
#ax.plot([i for i in range(len(final))],[i for i,j in final])

ax.errorbar([i for i in range(len(final))],[i for i,j in final],[j for i,j in final],marker="o",capsize=5)
# 
# 
y = np.array([i for i,j in final])
y_err = np.array([j for i,j in final])
x = [i for i in range(len(final))]
# 
ax.fill_between(x, y - y_err, y + y_err,facecolor="blue",alpha=0.2)
#
#
ax.scatter([i for i in range(len(p_true))],p_true,color="red")
ax.grid()
plt.show()
#
# 
# =============================================================================

P_even = np.array(final[::2])

P_true_even = np.array(p_true[::2])



# =============================================================================
# 
fig, ax = plt.subplots(figsize=(8,8))
# 
#ax.plot([i for i in range(len(final))],[i for i,j in final])

ax.errorbar([i for i in range(len(P_even))],[i for i,j in P_even],[j for i,j in P_even],marker="o",capsize=5)
# 
# 
y = np.array([i for i,j in P_even])
y_err = np.array([j for i,j in P_even])
x = [i for i in range(len(P_even))]
# 
ax.fill_between(x, y - y_err, y + y_err,facecolor="blue",alpha=0.2)
# 
#
ax.plot([i for i in range(len(P_true_even))],P_true_even,color="red")
ax.scatter([i for i in range(len(P_true_even))],P_true_even,color="red")
ax.grid()
plt.show()
# =============================================================================
