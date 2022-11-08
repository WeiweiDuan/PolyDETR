import os, cv2, math, sys
import numpy as np
import networkx as nx
import copy

def construct_sub_graph(map_nodes_dict, map_edges_list, boundary):
    #input: boundary: [xmin, ymin, xmax, ymax]
    #return: sub_node_dict={1: [x, y]}, sub_edges_list=[(1,2), (2,3)]
    xmin, ymin, xmax, ymax = boundary
    sub_node_dict = {}
    for c, node in map_nodes_dict.items():
        x, y = node
        if x > xmin and x < xmax and y > ymin and y < ymax:
            sub_node_dict[c] = [node[0]-xmin, node[1]-ymin]
    
    sub_edges_list = []
    for node_a_id in sub_node_dict.keys():
        for node_b_id in sub_node_dict.keys():
            if (node_a_id, node_b_id) in map_edges_list \
                or (node_b_id, node_a_id) in map_edges_list:
                sub_edges_list.append((node_a_id, node_b_id))
    return sub_node_dict, sub_edges_list

def raster_edges(node_ind, sub_nodes_dict, sub_edges_list, img_size=256):
    # input: the node idx, and the edges
    # output: a raster image, in which is the polyline that the node belongs to
    raster_polyline = np.zeros((img_size, img_size))
    for s_ind, e_ind in sub_edges_list:
        if s_ind == node_ind or e_ind == node_ind:
            x1, y1 = sub_nodes_dict[s_ind]
            x2, y2 = sub_nodes_dict[e_ind]
            cv2.line(raster_polyline, (y1, x1), (y2, x2), 1, 1)
    return raster_polyline

def gen_node_cls_reg_targets(sub_nodes_dict, sub_edges_list, gsize = 16, img_size=256):
    stride = img_size // gsize
    
    node_cat_targets = np.zeros(stride**2)
    node_reg_targets = np.zeros((stride**2, 2))
    nodes_in_grids = np.zeros((stride**2, 2))
    rm_nodes_list = []
    
    for i, node in sub_nodes_dict.items():
        x, y = node
        grid_x, grid_y = x//gsize, y//gsize
        
        raster_grid \
            = raster_edges(i, sub_nodes_dict, sub_edges_list, img_size=img_size)[grid_x*gsize:(grid_x+1)*gsize, grid_y*gsize:(grid_y+1)*gsize]
        nonzero_indices = np.where(raster_grid!=0)
#         x_in_grid, y_in_grid = int(x - grid_x*gsize), int(y - grid_y*gsize)
        
        if node_cat_targets[grid_x*stride+grid_y] == 1:
            # update the sub_graph, i.e., sub_nodes_dict, sub_edges_list
            # rm the node from sub_nodes_dict
            rm_nodes_list.append(i)
        
        elif nonzero_indices[0].size == 0:
            rm_nodes_list.append(i)
            
        else:
            avg_x, avg_y = np.mean(nonzero_indices[0]), np.mean(nonzero_indices[1])
            avg_x_in_img, avg_y_in_img = int(avg_x + grid_x*gsize), int(avg_y + grid_y*gsize)
            node_cat_targets[grid_x*stride+grid_y] = 1
            node_reg_targets[grid_x*stride+grid_y] = [avg_x, avg_y]
            nodes_in_grids[grid_x*stride+grid_y] = [avg_x_in_img, avg_y_in_img]
            ##### update the node_dict
            sub_nodes_dict[i] = [avg_x_in_img, avg_y_in_img]
    
    for i in rm_nodes_list:
        removed_value = sub_nodes_dict.pop(i)
        adj_nodes = []
        rm_edges = []
        for n1, n2 in sub_edges_list:
            if n1 == n2 and n1 == i:
                rm_edges.append((n1,n2))
            elif n1 == i:      
                rm_edges.append((n1,n2))
#                 if n2 not in rm_nodes_list:
                adj_nodes.append(n2)
            elif n2 == i:
                rm_edges.append((n1,n2))
#                 if n1 not in rm_nodes_list:
                adj_nodes.append(n1)
        
        for e in rm_edges:
            sub_edges_list.remove(e)
        adj_nodes = list(set(adj_nodes))

        for j in range(len(adj_nodes)):
            for k in range(j+1, len(adj_nodes)):
                sub_edges_list.append((adj_nodes[j], adj_nodes[k]))
    
    if np.sum(node_cat_targets) == 0:
        num_rand_nodes = np.random.randint(2, 4)
        rand_dec_inputs = np.random.randint(0, img_size, (num_rand_nodes, 2))
        grid_xy = np.array([rand_dec_inputs[:, 0]//stride, rand_dec_inputs[:, 1]//stride])
        for i, rand_node in enumerate(rand_dec_inputs):
            nodes_in_grids[:i] = rand_node
            
    return node_cat_targets, node_reg_targets, nodes_in_grids, sub_nodes_dict, sub_edges_list

def gen_adj_conn_targets(nodes_in_grids, sub_nodes_dict, sub_edges_list, \
                         gsize=16, img_size=256, n_nodes=150):
    nodes_inputs = np.ones((n_nodes, 2))
    adj_targets = np.zeros((n_nodes, n_nodes))
    conn_targets = np.zeros((n_nodes, n_nodes))
    nodes_mask = np.ones((n_nodes))
    adj_idx = {} #{idx_in_nodes_inputs: [x, y]}
    counter = 0
    stride = img_size // gsize
    for s_id, e_id in sub_edges_list:
        x_s_shp, y_s_shp = sub_nodes_dict[s_id]
        x_e_shp, y_e_shp = sub_nodes_dict[e_id]
#         if math.sqrt((x_s_shp-x_e_shp)**2+(y_s_shp-y_e_shp)**2) > 16.0:
#             print([x_s_shp, y_s_shp], [x_e_shp, y_e_shp])
        
        s_grid_x, s_grid_y = x_s_shp//gsize, y_s_shp//gsize
        e_grid_x, e_grid_y = x_e_shp//gsize, y_e_shp//gsize
        
        
        x_s, y_s = nodes_in_grids[s_grid_x*stride+s_grid_y].tolist()
        x_e, y_e = nodes_in_grids[e_grid_x*stride+e_grid_y].tolist()
        
        if x_s + y_s == 0 or x_e + y_e == 0:
            continue
            
#         if math.sqrt((x_s-x_e)**2+(y_s-y_e)**2) > 50.0:
#             print('*********')
#             print([x_s_shp, y_s_shp], [x_e_shp, y_e_shp])
#             print([x_s, y_s], [x_e, y_e])
            
        
        if [x_s, y_s] not in nodes_inputs[:counter].tolist():
            nodes_inputs[counter] = [x_s, y_s]
            nodes_mask[counter] = 0
            adj_idx[counter] = [x_s, y_s]
            counter += 1
                
        if [x_e, y_e] not in nodes_inputs[:counter].tolist():
            nodes_inputs[counter] = [x_e, y_e]
            nodes_mask[counter] = 0
            adj_idx[counter] = [x_e, y_e]
            counter += 1
    
        s_in_adj = list(adj_idx.keys())[list(adj_idx.values()).index([x_s, y_s])]
        e_in_adj = list(adj_idx.keys())[list(adj_idx.values()).index([x_e, y_e])]
        adj_targets[s_in_adj, e_in_adj] = 1
        adj_targets[e_in_adj, s_in_adj] = 1
    
    num_nodes = len(adj_idx.keys())
    
    G = nx.from_numpy_matrix(adj_targets)
    for i in range(num_nodes):
        connected_nodes = nx.node_connected_component(G, i)
        for j in connected_nodes:
            if i == j:
                continue
            conn_targets[i][j] = 1 
    
    if len(sub_edges_list) == 0:
        c = 0
        for i, n in enumerate(nodes_in_grids):
            if np.sum(n) != 0:
                nodes_mask[c] = 0
                nodes_inputs[c] = n
                c += 1
                
    ##### add small shifting to the nodes_inputs
    rand_noise = np.random.randint(-3, 3, size=(num_nodes, 2))
    pad_rand_noise = np.zeros((n_nodes, 2))
    pad_rand_noise[:num_nodes, :] = rand_noise
    nodes_inputs_noise = np.clip(nodes_inputs + pad_rand_noise, 0, img_size-1)
    return nodes_inputs_noise, adj_targets, conn_targets, nodes_mask
    