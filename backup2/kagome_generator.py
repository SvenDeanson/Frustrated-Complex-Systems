import numpy as np
import pulp as plp
import random
import matplotlib.pyplot as plt
import time
import networkx as nx

def Create_Points(n,m,plot=False,base="hexagon"):
    
    a = 2
    
    #basis = np.array([[0,0],[a/2,0],[a/4,np.sqrt(3)/2],[0,np.sqrt(3)],[a/2,np.sqrt(3)]])
    #basis2 = np.array([[0,0],[a/2,0],[a/4,np.sqrt(3)/2]])
    #basis3 = np.array([[a/4,np.sqrt(3)/2],[0,np.sqrt(3)],[a/2,np.sqrt(3)]])

    hexagon_basis = np.array([[a/4,np.sqrt(3)/2],[a/2,np.sqrt(3)]
                    ,[a,np.sqrt(3)],[a,0],[a/2,0],
                    [5*1/4*a,np.sqrt(3)/2]])

    star_basis = np.array([[0,0],[a/2,0],[a/4,np.sqrt(3)/2],[0,np.sqrt(3)],[a/2,np.sqrt(3)],
                     [3/4*a,3/2*np.sqrt(3)],[a,np.sqrt(3)],[a,0],
                     [3/4*a,-np.sqrt(3)/2],
                     [3/2*a,0],[3/2*a,np.sqrt(3)],
                     [5*1/4*a,np.sqrt(3)/2]])
    
    if base == "hexagon":
        basis = hexagon_basis
    if base == "star":
        basis = star_basis
    
    vec = np.array([[a,0],[a/2,a*np.sqrt(3)/2]])
    
    rows=m
    cols=n
    """
    vectors = [(0,0)]
    
    vectors = []
    for i in range(0,n+1):
         vectors.append((i,0))
    for i in range(1,n+1):
         vectors.append((i,-1))     
    for i in range(0,n):
         vectors.append((i,1))
    """        

    points=[]
    
    vectors=[(j,i) for i in range(0,rows) for j in range(0,cols)]
    """
    vectors = [(0,-1),(1,-1),
              (0,0),(1,0),
              (0,1),(1,1)]
    """
    vectors = sorted(vectors,key=lambda l:l[1])
    
    #print(vectors)
    
    for v in vectors:
        point = basis + v[0]*vec[0,:] + v[1]*vec[1,:]
        points.append(point)  
     
    points=np.array(points)

    POINTS= []
    for v in points:
        for a,b, in v:
            POINTS.append([a,b])

    lista = np.unique(np.round(POINTS,7), axis=0)
          
    #############################################################################
    
    if plot==True:
        fig, ax = plt.subplots(figsize=(10,10))
        x, y = np.array(lista).T

        ax.scatter(x,y)
        
        for i in range(-100,100):
            plt.axline((i, 0), slope=np.sqrt(3), color="black",alpha=0.3,linestyle='--', linewidth='1') #linestyle=(0, (5, 5)))
        for i in range(-100,100):
            plt.axline((i, 0), slope=-np.sqrt(3), color="black",alpha=0.3,linestyle='--', linewidth='1') #linestyle=(0, (5, 5)))
        for i in range(-100,100):
            plt.axline((0, np.sqrt(3)/2*i), slope=0,color="black",alpha=0.3,linestyle='--', linewidth='1')

        ax.set_xlim(xmin=min(x)-n,xmax=max(x)+n)
        ax.set_ylim(ymin=min(y)-n,ymax=max(y)+n)
        plt.grid(alpha=0.3)
        
        plt.show()
    
    #############################################################################
    
    return lista

def Create_Kagome_Graph(n,m,p=0.0,plot=True):
    
    points = Create_Points(n,m,plot=False,base="star")
    
    N = len(points)
    
    points = sorted(points,key=lambda l:l[1])
    
    ############################################################################
    G = nx.Graph()
    G.add_nodes_from([i for i in range(0,N)])
    
    dic={}  
    for n,(x,y) in zip(G.nodes(),points):
        dic[n] = (x,y)
    #############################################################################    
    for node in G.nodes():
        if (node)+1 in G.nodes():
            G.add_edge((node),node+1)

    remove=[]
    for (e1,e2),(index,(x,y)) in zip(G.edges(),enumerate(points)):
        try:
            if (points[index][1] != points[index+1][1]) or np.sqrt((points[index+1][0] - points[index][0])**2 + (points[index+1][1] - points[index+1][1])**2)>1:
                remove.append((e1,e2))
        except IndexError:
            print("do what needs to be done in this case")
    
    G.remove_edges_from(remove)

    remove=[]
    maxs = max((np.count_nonzero(points == y, axis=0))[1] for x,y in points)
    for node in G.nodes():
        for i in range(2,maxs+1):
            if (node)+i in G.nodes():
                if np.round(np.sqrt((dic[node+i][0] - dic[node][0])**2 + (dic[node+i][1] - dic[node][1])**2),2) <= 1:#np.sqrt(3)/4:
                    G.add_edge((node),node+i)
                    
    #####################################################################################                

    nukem=[]
    addem=[]
    for node in G.nodes():
        for edge in nx.edges(G,node):
            r = random.uniform(0, 1)
            if r<=p:
                nukem.append(edge)
                    
                new_node =(random.randint(0,N-1))    
                
                if new_node in G.nodes() and new_node!=node:
                    addem.append((node,new_node))
                else:
                    while new_node not in G.nodes() and new_node!=node:
                        new_node =(random.randint(0,N-1))  
                addem.append((node,new_node))
            
    G.remove_edges_from(nukem)
    G.add_edges_from(addem)
    
    ###########################################################################################   
    if plot == True:
        pos = {k: v for k,v in zip(G.nodes(), points)}
        fig, ax = plt.subplots(figsize=(8,8))
        nx.draw_networkx(G,pos=pos,node_size=1,with_labels=False,node_color="red")
    ###################################################################################### 
    
    return G
