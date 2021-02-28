# coding: utf-8

#read dat files using readlines and
#create lists for the 3 datasets representing x, y coordinates of the customers
#as well as the optimal sequences
coord = [[]]*3
for i, j in enumerate(['eil51', 'eil76', 'eil101']):
    fname = r'./data/'+j+'.dat'
    with open(fname) as f:
        content = f.readlines()
    coord[i] = [list(map(int, x.strip().split()))[1:] for x in content]

opt = [[]]*3
for i, j in enumerate(['eil51opt', 'eil76opt', 'eil101opt']):
    fname =  r'./data/'+j+'.dat'
    with open(fname) as f:
        content = f.readlines()
    opt[i] = [x.strip() for x in content]
    if i == 2: #eil101opt file has an empty line at the end of the file
        opt[i] = opt[i][:-1]
    opt[i] = [int(x)-1 for x in opt[i]] #city numbers are reduced by 1, because python indexing start from 0

#define a function for calculating tour distance given sequence
from scipy.spatial import distance, distance_matrix

def tourLen(seq, dist):
    frm = seq
    to = seq[1:] + [seq[0]]
    return sum(dist[[frm, to]])

#distance matrices for 51, 76 and 101 cities
dist_mat_51 = distance_matrix(coord[0], coord[0], p=2)
dist_mat_76 = distance_matrix(coord[1], coord[1], p=2)
dist_mat_101 = distance_matrix(coord[2], coord[2], p=2)
#optimal tour length of 51, 76 and 101 cities
tourLen(opt[0], dist_mat_51), tourLen(opt[1], dist_mat_76), tourLen(opt[2], dist_mat_101)


# # ONE SIDED NN

import numpy as np

def nearest_neighbor_1(n, dist_mat, kind):
    vertices = [[]] * 3
    for idx, init in enumerate([10, 20, 30]):
        vertices[idx] = [init - 1]  # index 9 represents city 10 since python indices start from 0
        while len(vertices[idx]) <= n-1:
            neigh = dist_mat[vertices[idx][-1], :].argsort(kind=kind)
            neighbors = [x for x in neigh if x not in vertices[idx]] #find neighbors that have not been selected yet
            vertices[idx].append(neighbors[0])
    return vertices

#nearest_neighbor_1(101,dist_mat_101)
print('### nearest neighbor 1')
seq_51 = nearest_neighbor_1(51,dist_mat_51, 'mergesort')
print(tourLen(seq_51[0],dist_mat_51) , tourLen(seq_51[1],dist_mat_51), tourLen(seq_51[2],dist_mat_51))
seq_76 = nearest_neighbor_1(76,dist_mat_76, 'quicksort')
print(tourLen(seq_76[0],dist_mat_76) , tourLen(seq_76[1],dist_mat_76), tourLen(seq_76[2],dist_mat_76))
seq_101 = nearest_neighbor_1(101,dist_mat_101, 'mergesort')
print(tourLen(seq_101[0],dist_mat_101) , tourLen(seq_101[1],dist_mat_101), tourLen(seq_101[2],dist_mat_101))


# # TWO SIDED NN

def nearest_neighbor_2(n, dist_mat, kind):
    vertices = [[]] * 3
    for idx, init in enumerate([10, 20, 30]):
        vertices[idx] = [init - 1]  # index 9 represents city 10 since python indices start from 0
        while len(vertices[idx]) <= n - 1:
            neigh_beg = dist_mat[vertices[idx][0], :].argsort(kind=kind) #neighbors to tour start
            neigh_end = dist_mat[vertices[idx][-1], :].argsort(kind=kind) #neighbors to tour end

            neighbors_beg = [x for x in neigh_beg if x not in vertices[idx]][0]
            neighbors_end = [x for x in neigh_end if x not in vertices[idx]][0]

            #select the minimum distance neighbor to tour start and end
            if dist_mat[vertices[idx][0], neighbors_beg] <= dist_mat[vertices[idx][-1], neighbors_end]:
                selected = neighbors_beg
                vertices[idx].insert(0, selected)
            else:
                selected = neighbors_end
                vertices[idx].append(selected)
    return vertices
print('### nearest neighbor 2')
seq_51 = nearest_neighbor_2(51, dist_mat_51, 'mergesort')
print(tourLen(seq_51[0], dist_mat_51), tourLen(seq_51[1], dist_mat_51), tourLen(seq_51[2], dist_mat_51))
seq_76 = nearest_neighbor_2(76, dist_mat_76, 'quicksort')
print(tourLen(seq_76[0], dist_mat_76), tourLen(seq_76[1], dist_mat_76), tourLen(seq_76[2], dist_mat_76))
seq_101 = nearest_neighbor_2(101, dist_mat_101, 'mergesort')
print(tourLen(seq_101[0], dist_mat_101), tourLen(seq_101[1], dist_mat_101), tourLen(seq_101[2], dist_mat_101))


# # NEAREST INSERT

vertices=[[]]*3

def nearest_insert(n, dist_mat, kind): #n : number of cities, dist_mat: distance matrix for cities
    vertices=[[]]*3
    for idx, init in enumerate([10, 20, 30]):
        vertices[idx]=[init-1]
        while len(vertices[idx])<=n-1:
            nearest=[]
            near_len=[]
            #selection step
            for i,j in enumerate(vertices[idx]):
                candid = dist_mat[j,:].argsort(kind=kind) #candidate cities for selection
                #select the nearest candidate that is not already in the tour
                nearest.append([x for x in candid if x not in vertices[idx]][0])
                near_len.append(dist_mat[j,nearest[i]])
            selected=nearest[np.argmin(near_len)]
            if len(vertices[idx])<=2:
                vertices[idx].append(selected)
                continue
            #insertion step
            temp_vertices = vertices[idx]+[vertices[idx][0]]
            arc_candid = []
            #find all possible insertions and their costs=(increase in distance)
            for a in range(len(temp_vertices)-1):
                c_ij=dist_mat[temp_vertices[a], temp_vertices[a+1]]
                c_ik=dist_mat[temp_vertices[a], selected]
                c_kj=dist_mat[selected, temp_vertices[a+1]]
                arc_candid.append(c_ik+c_kj-c_ij)
            index = np.argmin(arc_candid)+1
            vertices[idx].insert(index, selected)
    return vertices
print('### nearest insert')
seq_51 = nearest_insert(51, dist_mat_51, 'mergesort')
print(tourLen(seq_51[0], dist_mat_51), tourLen(seq_51[1], dist_mat_51), tourLen(seq_51[2], dist_mat_51))
seq_76 = nearest_insert(76, dist_mat_76, 'quicksort')
print(tourLen(seq_76[0], dist_mat_76), tourLen(seq_76[1], dist_mat_76), tourLen(seq_76[2], dist_mat_76))
seq_101 = nearest_insert(101, dist_mat_101, 'mergesort')
print(tourLen(seq_101[0], dist_mat_101), tourLen(seq_101[1], dist_mat_101), tourLen(seq_101[2], dist_mat_101))


# # FARTHEST INSERT

def farthest_insert(n, dist_mat, sort): #n : number of cities, dist_mat: distance matrix for cities
    vertices=[[]]*3
    for idx, init in enumerate([10, 20, 30]):
        vertices[idx] = [init - 1]  # index 9 represents city 10 since python indices start from 0
        while len(vertices[idx]) <= n-1:
            farthest = []
            near_len = []
           # selection step
            for i, j in enumerate(vertices[idx]):
                candid = (-dist_mat[j, :]).argsort(kind=sort) #candidate cities for selection, distances are sorted descendingly
                #select the farthest candidate that is not already in the tour
                farthest.append([x for x in candid if x not in vertices[idx]][0])
                near_len.append(dist_mat[j, farthest[i]])
            selected = farthest[np.argmax(near_len)]
            if len(vertices[idx]) <= 2:
                vertices[idx].append(selected)
                continue
            # insertion step
            temp_vertices = vertices[idx] + [vertices[idx][0]]
            arc_candid = []
            #find all possible insertions and their costs=(increase in distance)
            for a in range(len(temp_vertices) - 1):
                c_ij = dist_mat[temp_vertices[a], temp_vertices[a + 1]]
                c_ik = dist_mat[temp_vertices[a], selected]
                c_kj = dist_mat[selected, temp_vertices[a + 1]]
                arc_candid.append(c_ik + c_kj - c_ij)
            index = np.argmin(arc_candid) + 1
            vertices[idx].insert(index, selected)
    return vertices
print('### farthest insert')
seq_51 = farthest_insert(51, dist_mat_51, 'mergesort')
print(tourLen(seq_51[0], dist_mat_51), tourLen(seq_51[1], dist_mat_51), tourLen(seq_51[2], dist_mat_51))
seq_76 = farthest_insert(76, dist_mat_76, 'mergesort')
print(tourLen(seq_76[0], dist_mat_76), tourLen(seq_76[1], dist_mat_76), tourLen(seq_76[2], dist_mat_76))
seq_101 = farthest_insert(101, dist_mat_101, 'mergesort')
print(tourLen(seq_101[0], dist_mat_101), tourLen(seq_101[1], dist_mat_101), tourLen(seq_101[2], dist_mat_101))


# # 2-OPT

def two_opt(seq, dist_mat): #n : number of cities, dist_mat: distance matrix for cities
    n=len(seq)
    delta_min = -1
    #continue until there is no improvement in tour length
    while delta_min < 0:
        delta = []
        arcs = []
        #find all possible arcs that can be replaced
        for i in range(0, n):
            for k in range(i + 2, n):
                j = (i + 1) % n
                k = k % n
                l = (k + 1) % n
                #cycling is prevented by following if conditions
                if i == l:
                    continue
                if j == k:
                    continue
                d_ik = dist_mat[seq[i],seq[k]]
                d_jl = dist_mat[seq[j],seq[l]]
                d_ij = dist_mat[seq[i],seq[j]]
                d_kl = dist_mat[seq[k],seq[l]]
                delta_current=d_ik+d_jl-d_ij-d_kl
                delta.append(delta_current)
                arcs.append([i,j,k,l])
        delta_min = min(delta)
        delta_min_i = np.argmin(delta)
        arc_min = arcs[delta_min_i]
        #for the best improvement swap nodes
        if delta_min < 0:
            seq[arc_min[1]] , seq[arc_min[2]] = seq[arc_min[2]] , seq[arc_min[1]]
    return seq


# In[9]:

#2_OPTS
#One-sided NN
print('### one sided nearest neighbor 2-opt')
seq_51=nearest_neighbor_1(51,dist_mat_51, 'mergesort')
seq_51_2 = [two_opt(x, dist_mat_51) for x in seq_51]
print(tourLen(seq_51_2[0], dist_mat_51), tourLen(seq_51_2[1], dist_mat_51), tourLen(seq_51_2[2], dist_mat_51))
seq_76_2 = [two_opt(x, dist_mat_76) for x in nearest_neighbor_1(76,dist_mat_76, 'quicksort')]
print(tourLen(seq_76_2[0], dist_mat_76), tourLen(seq_76_2[1], dist_mat_76), tourLen(seq_76_2[2], dist_mat_76))
seq_101_2 = [two_opt(x, dist_mat_101) for x in nearest_neighbor_1(101,dist_mat_101, 'mergesort')]
print(tourLen(seq_101_2[0], dist_mat_101), tourLen(seq_101_2[1], dist_mat_101), tourLen(seq_101_2[2], dist_mat_101))


#2_OPTS
#Two-sided NN
print('### 2 sided nearest neighbor 2-opt')
seq_51=nearest_neighbor_2(51,dist_mat_51, 'quicksort')
seq_51_2 = [two_opt(x, dist_mat_51) for x in seq_51]
print(tourLen(seq_51_2[0], dist_mat_51), tourLen(seq_51_2[1], dist_mat_51), tourLen(seq_51_2[2], dist_mat_51))
seq_76_2 = [two_opt(x, dist_mat_76) for x in nearest_neighbor_2(76,dist_mat_76, 'quicksort')]
print(tourLen(seq_76_2[0], dist_mat_76), tourLen(seq_76_2[1], dist_mat_76), tourLen(seq_76_2[2], dist_mat_76))
seq_101_2 = [two_opt(x, dist_mat_101) for x in nearest_neighbor_2(101,dist_mat_101, 'quicksort' )]
print(tourLen(seq_101_2[0], dist_mat_101), tourLen(seq_101_2[1], dist_mat_101), tourLen(seq_101_2[2], dist_mat_101))


# 2_OPTS
#Nearest Insert
print('### nearest insert 2-opt')
seq_51_2 = [two_opt(x, dist_mat_51) for x in nearest_insert(51,dist_mat_51, 'quicksort')]
print(tourLen(seq_51_2[0], dist_mat_51), tourLen(seq_51_2[1], dist_mat_51), tourLen(seq_51_2[2], dist_mat_51))
seq_76_2 = [two_opt(x, dist_mat_76) for x in nearest_insert(76,dist_mat_76, 'mergesort')]
print(tourLen(seq_76_2[0], dist_mat_76), tourLen(seq_76_2[1], dist_mat_76), tourLen(seq_76_2[2], dist_mat_76))
seq_101_2 = [two_opt(x, dist_mat_101) for x in nearest_insert(101,dist_mat_101, 'quicksort')]
print(tourLen(seq_101_2[0], dist_mat_101), tourLen(seq_101_2[1], dist_mat_101), tourLen(seq_101_2[2], dist_mat_101))


#2_OPTS
#Farthest Insert
print('### farthest insert 2-opt')
seq_51_2 = [two_opt(x, dist_mat_51) for x in farthest_insert(51,dist_mat_51, 'quicksort')]
print(tourLen(seq_51_2[0], dist_mat_51), tourLen(seq_51_2[1], dist_mat_51), tourLen(seq_51_2[2], dist_mat_51))
seq_76_2 = [two_opt(x, dist_mat_76) for x in farthest_insert(76,dist_mat_76, 'quicksort')]
print(tourLen(seq_76_2[0], dist_mat_76), tourLen(seq_76_2[1], dist_mat_76), tourLen(seq_76_2[2], dist_mat_76))
seq_101_2 = [two_opt(x, dist_mat_101) for x in farthest_insert(101,dist_mat_101, 'quicksort')]
print(tourLen(seq_101_2[0], dist_mat_101), tourLen(seq_101_2[1], dist_mat_101), tourLen(seq_101_2[2], dist_mat_101))
