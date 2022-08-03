#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 12:33:26 2022

@author: sven
"""

from all_functions import *


"""
G = Create_Minimum_Data(10,2)
Plot2D(G,nx.spring_layout(G, weight=None),5)

######################################################################
edge_length = len(G.edges())
print("L = ",edge_length)
print("f =",frustration_count(G),"d =",calculate_delta(G))
p,n = color_of_frustration(G)
print("p =",p,"n =",n)
######################################################################

print("success!")
"""

def Test_Minimal(N,choice,timesteps=100):
    
    g0 = Create_Minimum_Data(N,choice)
    
    edge_length = len(g0.edges())
    
    d = calculate_delta(g0)
    f = frustration_count(g0)
    
    print("\nL = ",edge_length,"d0 = ",d,"f0 = ",f,"\n")
    
    Gs = [g0]
    
    start_time = time.time()
    
    data = Quench(Gs,timesteps)
    
    #print("\nQuench time:",time.time() - start_time) 
    
    #avg_P,P_even,avg_M = Extract_Information(data,edge_length,timesteps)
    start_time = time.time()
    
    P_t,P_even =  Get_Probabilites0(data,edge_length,timesteps)
    Means = Get_Means(data[0])
    
    #print("\nProbability and Means: ",time.time() - start_time) 
    
    f0 = f
    d0 = d
    
    lmbd_sol = analytical(edge_length,f0,d0)
        
    xs = np.linspace(0,timesteps-1,timesteps)
    #print(np.round(xs))
    y_actual= lmbd_sol(xs)
    
    ####################################################################################################
    """
    fig, ax = plt.subplots(figsize=(4,4))
    
    x  = np.array([i for i in range(len(Means))])
    y = np.array([i for i,j in Means])
    y_err = np.array([j for i,j in Means])
    
    ax.plot(x,y)
    ax.plot(xs,y_actual)
    ax.fill_between(x, y - y_err, y + y_err,facecolor="blue",alpha=0.2)
    ax.set_ylim(0,edge_length)
    ax.grid()
    plt.show()
    """
    ####################################################################################################
    
    
    from sklearn.metrics import mean_squared_error
    
    rms = np.sqrt(mean_squared_error(y_actual, [i for i,j in Means]))
    
    print("rms:",rms)

    return edge_length,rms, y_actual, Means

def Test_Random(N,choice,timesteps=100):
    
    g0 = Create_Random_Data(N,choice)
    
    edge_length = len(g0.edges())
    
    d = calculate_delta(g0)
    f = frustration_count(g0)
    
    print("\nL = ",edge_length,"d0 = ",d,"f0 = ",f,"\n")
    
    Gs = [g0]
    
    start_time = time.time()
    
    data = Quench(Gs,timesteps)
    
    #print("\nQuench time:",time.time() - start_time) 
    
    #avg_P,P_even,avg_M = Extract_Information(data,edge_length,timesteps)
    start_time = time.time()
    
    P_t,P_even =  Get_Probabilites0(data,edge_length,timesteps)
    Means = Get_Means(data[0])
    
    #print("\nProbability and Means: ",time.time() - start_time) 
    
    f0 = f
    d0 = d
    
    lmbd_sol = analytical(edge_length,f0,d0)
        
    xs = np.linspace(0,timesteps-1,timesteps)
    #print(np.round(xs))
    y_actual= lmbd_sol(xs)
    
    from sklearn.metrics import mean_squared_error
    
    rms = np.sqrt(mean_squared_error(y_actual, [i for i,j in Means]))
    
    print("rms:",rms)

    return edge_length,rms, y_actual, Means
