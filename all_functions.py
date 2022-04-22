# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Packages
###############################################################################
import time
import math
import random
import numpy as np
import pulp as plp
from tqdm import tqdm
import networkx as nx
from sympy import *
from gurobipy import *
import multiprocessing
from random import choice
from random import sample
import numpy.random as rn
from collections import Counter
import matplotlib.pyplot as plt
############################################################################################################################


from kagome_generator import *

def Optimization_Model(weights, signed_matrix):

    objectivevalue=[]
    objs=[]
    solveTime=[]
    
    index=0 
    order=len(signed_matrix[index])

    ###############################################################################################
    
    opt_model = plp.LpProblem(name="Binary_Model",sense = plp.LpMinimize)


    x=[]
    for i in range(0,order):
        x.append(plp.LpVariable(lowBound=0,upBound=1, cat=plp.LpBinary, name='x'+str(i)))
    z={}    
    for (i,j) in (weights[index]):
        z[(i,j)]=plp.LpVariable(lowBound=0,upBound=1, cat=plp.LpBinary, name='z'+str(i)+','+str(j))
    ###############################################################################################
    OFV = 0
    for (i,j) in (weights[index]):
        OFV += z[(i,j)]

    opt_model.setObjective(OFV)

    for (i,j) in (weights[index]):
        opt_model.addConstraint( z[(i,j)] >= x[i] - ((weights[index])[(i,j)])*x[j] -\
                        (1-(weights[index])[(i,j)])/2)
        opt_model.addConstraint( z[(i,j)] >= -x[i] + ((weights[index])[(i,j)])*x[j] +\
                        (1-(weights[index])[(i,j)])/2)   

    ###############################################################################################
    
    start_time = time.time()
    status = opt_model.solve(solver = plp.getSolver('GUROBI_CMD',msg=0))#'COIN_CMD'))
    solveTime.append(time.time() - start_time) 

    ###############################################################################################

    varsdict = {}
    for v in opt_model.variables():
        varsdict[v.name] = v.varValue    
                    
    return float(opt_model.objective.value()), varsdict

###################################################################################################################################
def Prepare_Data(matrices):

    signed_matrices=[]
    unsigned_matrices=[]
    weights = []
    graphs=[]
    for signed_matrix in matrices:

        Graph=nx.from_numpy_matrix(signed_matrix)

        mapping=dict(zip(Graph.nodes(),range(len(Graph.nodes()))))
        Graph=nx.relabel_nodes(Graph,mapping) 

        graphs.append(Graph)

        signed_matrix = nx.to_numpy_matrix(Graph)
        unsigned_matrix = abs(signed_matrix)    

        weighted_edges=nx.get_edge_attributes(Graph, 'weight') 
        w={}
        for (u,v) in weighted_edges:
            if u<v:
                w[(u,v)] = weighted_edges[(u,v)]
            if u>v:
                w[(v,u)] = weighted_edges[(u,v)]

        signed_matrices.append(signed_matrix)
        unsigned_matrices.append(unsigned_matrix)
        weights.append(w)

    return graphs, weights, signed_matrices


def color_nodes(G,variables):
    
    d = {}
    for key, value in variables.items():
        if "x" in key:
            #print(key[1:],value)
            d[int(key[1:])]=value

    var = dict(sorted(d.items()))
    
    node_colors=[]
    for key,value in var.items():
        if value==1:
            node_colors.append("Black")
        else:
            node_colors.append("Silver")
            
    attr = {}
    for (node,value),color in zip(G.nodes.data(),node_colors):
        #print(node,color)
        attr[node]=color
        
    nx.set_node_attributes(G, attr, 'color')
    
    return G

def Add_Weights(H,k):
    
    N = len(H.edges())

    m=int(round(k*N))
    n=int(round((1-k)*N))

    a = np.ones(n+m)
    a[:m] = -1
    np.random.shuffle(a)
    
    dic={}
    for e,value in zip(H.edges(),a):
        dic[e] = value
    
    nx.set_edge_attributes(H, dic,'weight')
    
    return H

###################################################################################################################################

def Create_Minimum_Data(n,num):
    
    k = 0.5
    
    if num==1:
        G = nx.grid_2d_graph(n,n)
    if num==2:
        G = nx.triangular_lattice_graph(n,n)
    if num==3:
        G= nx.path_graph(n)
    if num==4:
        G = Create_Triangular_Graph(n)
    if num==5:
        G =  Create_Kagome_Graph(n,n,plot=False)
        

    N = len(G.nodes())
    
    m=int(round(k*N))
    n=int(round((1-k)*N))

    a = np.ones(N)
    a[:m] = -1
    np.random.shuffle(a)
    
    node_colors=[]
    for i in a:
        if i==1:
            node_colors.append("Silver")
        else:
            node_colors.append("Black")

    attr = {}
    for (node,value),color in zip(G.nodes.data(),node_colors):
        attr[node]=color

    nx.set_node_attributes(G, attr, 'color')
    
    G = Add_Weights(G,1.0)
    
    
    matrix = nx.to_numpy_matrix(G)
    g, weights, signed_matrix = Prepare_Data([matrix])

    fi,vr = Optimization_Model(weights, signed_matrix)
    #frustrations.append(fi)
    G = color_nodes(G,vr)
    
    #pos = nx.spring_layout(G, weight=None)
    #nx.set_node_attributes(G,pos,'pos')
    
    return G

###################################################################################################################################
def Create_Triangular_Graph(n):
    p=[]
    for j in range(1,n+1):
        for i in range(1,n+1):
            p.append((j%2 + 2*i - 3, 2*j-2))
    
    points = p
    N = len(points)
    points = sorted(points,key=lambda l:l[1])
    
    g = nx.Graph()
    
    g.add_nodes_from(points)
    
    dic={}  
    for n,(x,y) in zip(g.nodes(),points):
        dic[n] = (x,y)

    for a,b in g.nodes():
        if (a+1,b+2) in g.nodes():
            g.add_edge((a,b),(a+1,b+2))
        if (a-1,b+2) in g.nodes():
            g.add_edge((a,b),(a-1,b+2))
        if (a+2,b) in g.nodes():
            g.add_edge((a,b),(a+2,b))

    return g



######################################################################################################
def Create_Random_Data(n,choice,k=False):
    
    if choice==1:
        G = nx.grid_2d_graph(n,n)
    if choice==2:
        G = nx.triangular_lattice_graph(n,n)
    if choice==3:
        G = nx.path_graph(n)
    if choice==4:
        G = Create_Triangular_Graph(n)
    if choice==5:
        G =  Create_Kagome_Graph(n,n,plot=False)


    N = len(G.nodes())
    
    if type(k) == bool:
        m = random.randrange(0,N)
        n = N-m
    else:
        m=int(round(k*N))
        n=int(round((1-k)*N))

    a = np.ones(N)
    a[:m] = -1
    np.random.shuffle(a)
    
    node_colors=[]
    for i in a:
        if i==1:
            node_colors.append("Silver")
        else:
            node_colors.append("Black")

    attr = {}
    for (node,value),color in zip(G.nodes.data(),node_colors):
        attr[node]=color

    nx.set_node_attributes(G, attr, 'color')
    
    G = Add_Weights(G,1.0)
    
    #pos = nx.spring_layout(G, weight=None)
    #nx.set_node_attributes(G,pos,'pos')
    
    return G

###################################################################################################################################

def Plot2D(G,pos,size=5):

    H0 = G.copy()
    #pos =nx.get_node_attributes(H0,'pos')
    node_colors=nx.get_node_attributes(H0,'color')

    edge_colors = []
    for n1,n2,val in H0.edges.data():

        #print(n1,n2)
        if H0.nodes[n1]['color']==H0.nodes[n2]['color']:

            if H0.nodes[n1]['color'] ==H0.nodes[n2]['color'] == "Black":
                edge_colors.append("red")
                #negative +=1
            if H0.nodes[n1]['color'] ==H0.nodes[n2]['color'] == "Silver":
                edge_colors.append("green")
                #positive +=1
        else:
                edge_colors.append("lightgrey")

    attr2 = {}
    for edge,color in zip(H0.edges(),edge_colors):

        #print(edge,color)
        attr2[edge]=color

    nx.set_edge_attributes(H0, attr2, 'edge_color')
    
    edge_colors=nx.get_edge_attributes(H0,'edge_color')
        
    #pos = nx.spring_layout(G, weight=None)
    
    options = {"node_size": 50, "alpha": 0.6}

    fig, ax = plt.subplots(figsize=(size,size))
    nx.draw(H0,pos,edge_color=edge_colors.values(),node_color=node_colors.values(),with_labels=False, width = 2.5,**options)
    
    plt.savefig('figure1.png')
    
    
###############################################################################
def frustration_count(G):
    s = 0
    for n1,n2,val in G.edges.data():
        if G.nodes[n1]['color']==G.nodes[n2]['color']:
            s+=1
    return s

def color_of_frustration(H):
    
    positive=0
    negative=0

    for n1,n2,val in H.edges.data():
        
        G = H.copy() 
        color1 = G.nodes[n1]['color']
        color2 = G.nodes[n2]['color']
        if color1 == color2 or color2==color1:
            #print("true")
            if color1 == color2 == "Black":
                negative +=1
            if color1 == color2 == "Silver":
                positive +=1
                
    return positive,negative

def calculate_delta(G):
    
    p,n = color_of_frustration(G)
    
    if p>n:
        return int(np.sqrt((p-n)**2))
    else:
        return int(np.sqrt((n-p)**2))
    
def analytical(L,f0,d0):

    edge = L

    f0 = f0

    t = symbols('t')
    x = symbols('x', cls=Function)
    L = symbols('L', real=True)
    d = symbols('d', real=True)

    gsol = dsolve(x(t).diff(t) - ((1 - 2*L)*x(t) + L*(L - 1) + d**2)/(L*(L - 1)), x(t),ics={x(0): f0})

    l=edge
    k=d0
    sol = gsol.subs({L: l,d:k})
    lmbd_sol = lambdify(t, sol.rhs)
    
    return lmbd_sol

###############################################################################


def Properties(G):
    positive,negative = color_of_frustration(G)
    
    print("N: ",len(G.nodes()))
    print("L: ",len(G.edges()))

    node_colors=nx.get_node_attributes(G,'color')
    #print("Black: ",len([i for i in node_colors.values() if i=="Black"]))
    #print("White: ",len([i for i in node_colors.values() if i=="Silver"])) 
    print("f: ",frustration_count(G))
    print("L-f: ",len(G.edges())-frustration_count(G))

    print("f+: ",positive)
    print("f-: ",negative)
    
    print("Delta: ",positive-negative)
    print("Abs(Delta): ",int(np.sqrt((positive-negative)**2)))

    print(frustration_count(G)/len(G.edges()))


###############ANNEALING#################

def random_neighbour(G):
 
    H = G.copy()
    
    random_node = sample(H.nodes(),1)[0]
    
    chosen_color = H.nodes[random_node]['color']
    if chosen_color == "Silver":
        H.nodes[random_node]['color'] = "Black"
    else:
        H.nodes[random_node]['color'] = "Silver"
    
    return H

def target_delta(state, target):
    
    new = calculate_delta(state)
    
    mse = np.abs(target-new)

    return mse

def acceptance_probability(cost, new_cost, temperature):
    if new_cost < cost:
        return 1
    else:
        p = np.exp(- (new_cost - cost) / temperature)
        return p

def temperature(fraction):
    
    value = max(0.01, min(1, 1 - fraction))
    return value

def Annealing(G0,target, maxsteps=1000):

    
    state = G0.copy()
    cost_funct = target_delta
    
    cost = cost_funct(state,target)
    
    if cost == 0:
        return state
    
    for step in range(maxsteps):
        
        fraction = step / float(maxsteps)
        T = temperature(fraction)
        
        new_state = random_neighbour(state)
        
        new_cost = cost_funct(new_state,target)
        
        if acceptance_probability(cost, new_cost, T) > rn.random():
            
            state, cost = new_state, new_cost
            if cost == 0:
                return state    
    return state

##############################################################################

###############QUENCH_PROCESS#################



def Quench_Process(Gs):

    total=[]
    Delta = []
    for i in (range(len(Gs))):
        
        H = Gs[i].copy()
        
        res=[]
        res.append(frustration_count(H))
        deltas = []
        deltas.append(calculate_delta(H))
        
        G = H.copy()
        for k in range(1,500):
            
            nx.algorithms.connected_double_edge_swap(G, nswap=1)
            
            fr = frustration_count(G)
            res.append(fr)
            deltas.append(calculate_delta(G))

        total.append(res) 
        Delta.append(deltas)
        
    return np.array(total),np.array(Delta)

def Quench(Gs,timesteps=100):
    

    total=[]
    
    for index,g0 in (enumerate(Gs)):
    
        specific_avg = []
        for i in (range(timesteps)):

            H = g0.copy()

            res=[]
            res.append(frustration_count(H))

            G = H.copy()
            
            #d0  = calculate_delta(G)
            for k in (range(1,timesteps)):

                nx.algorithms.connected_double_edge_swap(G, nswap=1)
                #G = nx.random_reference(G,connectivity=True)
    
                res.append(frustration_count(G))
        
                #if calculate_delta(G) != d0:
                    #print("Greška")

            specific_avg.append(res)
        
        total.append(specific_avg) 
            
    return np.array(total)

def Get_Probabilites0(data,Edge,repeat):

    p_t=[]
    for t in range(len(data[0])):

        prob=[]
        for i in range(0,Edge):
            prob.append(np.count_nonzero(np.array(data[:,t]) == i))
        p_t.append(np.array(prob)/repeat)

    P_t = np.array(p_t)

    p_even=[]
    for l in P_t:
        p_even.append(l[::2])

    P_even = np.array(p_even)

    return P_t,P_even

def Get_Probabilites(data,Edge,repeat):
    
    avg_P_t = []
    for a in data:
        p_t=[]
        for t in range(len(a[0])):

            prob=[]
            for i in range(0,Edge):
                prob.append(np.count_nonzero(np.array(a[:,t]) == i))
            p_t.append(np.array(prob)/repeat)

        P_t = np.array(p_t)
        avg_P_t.append(P_t)

    return np.array(avg_P_t)


def Get_Means(data):
    
    total = np.array(data)
    final = []
    for i in range(0,len(total[0])):
        final.append((np.mean(total[:,i]),np.std(total[:,i])))

    return np.array(final)

def Extract_Information(data,edge_length,timesteps):
    edge = edge_length

    avg_P_t = Get_Probabilites(data,edge,timesteps)

    
    avg_P = []
    for t in range(timesteps):
        avg_p = []
        for i in range(edge):
            avg_p.append((np.mean(avg_P_t[:,t,i]),np.std(avg_P_t[:,t,i])))
        avg_P.append(avg_p)
        
    avg_P = np.array(avg_P)
        
    p_even=[]
    for l in avg_P:
        p_even.append(l[::2])
    
    P_even = np.array(p_even)
    
    means_array = []
    for k in range(len(data)):
        means_k = Get_Means(data[k])

        means_array.append([i for i,j in means_k])


    avg_M = Get_Means(means_array)
    
    return avg_P,P_even,avg_M


def Load_Analytic_Solution(filepath):
    f = open(filepath,'r')

    p =[]
    for line in f:
        
        #p.append(float((line.rstrip("\n"))))
        if len(line.rstrip("\n"))==1:
                pappend(0)
        else:
            p.append(float((line.rstrip("\n")).replace("*^","e")))
    p = np.array(p)
    #p = np.delete(p, 0)
    p = np.delete(p, -1)
    
    return p
########################################################################################################################################################################