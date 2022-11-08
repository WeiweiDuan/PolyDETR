import torch
import cv2
import numpy as np

img_mean = [0.485, 0.456, 0.406]
img_std = [0.229, 0.224, 0.225]

def predict(model, x_torch_sub):
    with torch.no_grad():
        outputs, enc_attn, dec_attn = model.predict(x_torch_sub)
        
        node_cat_pred = torch.nn.functional.softmax(outputs['pred_cat_nodes'], dim=-1)
        node_cat_pred_np = node_cat_pred.cpu().detach().numpy()

        node_reg_pred = torch.nn.functional.sigmoid(outputs['pred_reg_nodes'])
        node_reg_pred_np = node_reg_pred.cpu().detach().numpy()

        adj_pred = torch.nn.functional.sigmoid(outputs['pred_adj'])
        adj_pred_np = adj_pred.cpu().detach().numpy()

        pred_dec_mask_np = outputs['mask'].cpu().detach().numpy()

        pred_dec_inputs_np = outputs['dec_inputs'].cpu().detach().numpy()
        
        for i in range(adj_pred_np.shape[1]):
            adj_pred_np[:, i, i] = 0
        enc_attn_np = enc_attn.cpu().detach().numpy()
        dec_attn_np = dec_attn.cpu().detach().numpy()
        
    return node_cat_pred_np, node_reg_pred_np, adj_pred_np, pred_dec_inputs_np, \
            pred_dec_mask_np, enc_attn_np, dec_attn_np

def construct_graph_prob(model, x_test_name, map_img, device, \
                         win_size=256, batch=32, thres=0.5, buffer=5, num_nodes_thres=1, dist_thres=40):
    ##### with flexible node match
    nodes_dict = {} # keys: nodes_id (start from 0), values: [x, y]
    edges_dict = {} #{(n_id, n_id): counter, ...}

    for i in range(0, len(x_test_name), batch):
        ##### process the input images
        sub_x_name = x_test_name[i:i+batch]
        x_test = []
        x_name_list = []
        for name in sub_x_name:
            row, col = name.split('_')
            row, col = int(row), int(col)
            img = map_img[row:row+win_size, col:col+win_size]
            if img.shape != (win_size, win_size, 3):
                del name
                continue
            x_test.append(img)
            x_name_list.append(name)

        if len(x_test) == 0:
            continue

        x_test_np = np.array(x_test)
        x_test_np = x_test_np.astype(np.float64)
        x_test_np  = x_test_np / (win_size-1)
        x_test_np  = (x_test_np  - img_mean) / img_std

        x_torch = torch.from_numpy(x_test_np.copy()).type(torch.FloatTensor).to(device)
        x_torch = torch.moveaxis(x_torch, 3, 1)

        ##### prediction
        node_cat_pred_np, node_reg_pred_np, adj_pred_np, pred_dec_inputs_np, pred_dec_mask_np,  \
           enc_attn_np, dec_attn_np = predict(model, x_torch)

        for img_idx in range(node_cat_pred_np.shape[0]):
            ####### get predicted nodes
            pred = np.argmax(node_cat_pred_np[img_idx,:,:], axis=-1)
            pred_node = node_reg_pred_np[img_idx,:,:]

            pos_idx = np.where(pred!=0)[0]
            if pos_idx.shape[0] < num_nodes_thres:
                continue

            row, col = x_name_list[img_idx].split('_')
            row, col = int(row), int(col)
#             if row % 100 == 0 or col % 100 ==0:
#                 print('===== processing %d row, %d col ====='%(row, col))
            ####### get predicted edges
            num_nodes = np.where(pred_dec_mask_np[img_idx]==0)[0].shape[0]
            adj_where = np.where(adj_pred_np[img_idx,:num_nodes,:num_nodes]>thres)

            for i, _ in enumerate(adj_where[0]):
                s, e = adj_where[0][i], adj_where[1][i]
                x1, y1 = (pred_dec_inputs_np[img_idx, s]*(win_size-1)).astype('int32')
                x2, y2 = (pred_dec_inputs_np[img_idx, e]*(win_size-1)).astype('int32')
                if (x1-x2)**2 + (y1-y2)**2 > dist_thres**2:
                    continue
                x1_in_map, y1_in_map = x1+row, y1+col
                x2_in_map, y2_in_map = x2+row, y2+col
                node1_in_dict, node2_in_dict  = False, False
                s_ind, e_ind = None, None
                for x_buff in range(-buffer, buffer, 1):
                    for y_buff in range(-buffer, buffer, 1):
                        if [x1_in_map+x_buff, y1_in_map+y_buff] in nodes_dict.values():
                            new_x1 = x1_in_map+x_buff//2
                            new_y1 = y1_in_map+y_buff//2
                            s_ind = list(nodes_dict.keys())[list(nodes_dict.values()).index([x1_in_map+x_buff, y1_in_map+y_buff])]
                            nodes_dict[s_ind] = [new_x1, new_y1]
                            node1_in_dict = True
                            break
                    if node1_in_dict:
                        break
                for x_buff in range(-buffer, buffer, 1):
                    for y_buff in range(-buffer, buffer, 1):
                        if [x2_in_map+x_buff, y2_in_map+y_buff] in nodes_dict.values():
                            new_x2, new_y2 = x2_in_map+x_buff//2, y2_in_map+y_buff//2
                            e_ind = list(nodes_dict.keys())[list(nodes_dict.values()).index([x2_in_map+x_buff, y2_in_map+y_buff])]
                            nodes_dict[e_ind] = [new_x2, new_y2]
                            node2_in_dict = True
                            break
                    if node2_in_dict:
                        break
                if not node1_in_dict:
                    s_ind = len(nodes_dict.keys())
                    nodes_dict[s_ind] = [x1_in_map, y1_in_map]
                if not node2_in_dict:
                    e_ind = len(nodes_dict.keys())
                    nodes_dict[e_ind] = [x2_in_map, y2_in_map]

                if (s_ind, e_ind) in edges_dict: 
                    edges_dict[(s_ind, e_ind)] += 1
                else:
                    edges_dict[(s_ind, e_ind)] = 1
                if(e_ind, s_ind) in edges_dict:
                    edges_dict[(e_ind, s_ind)] += 1
                else:
                    edges_dict[(e_ind, s_ind)] = 1
    return nodes_dict, edges_dict

def construct_graph_top2(model, x_test_name, map_img, device,\
                         win_size=256, batch=32, thres=0.5, buffer=5, num_nodes_thres=1, dist_thres=40):
    # ##### with flexible node match
    nodes_dict = {} # keys: nodes_id (start from 0), values: [x, y]
    edges_dict = {} #{(n_id, n_id): counter, ...}


    for i in range(0, len(x_test_name), batch):
        ##### process the input images
        sub_x_name = x_test_name[i:i+batch]
        x_test = []
        x_name_list = []
        for name in sub_x_name:
            row, col = name[:-4].split('_')
            row, col = int(row), int(col)
    #         if row < 7200 or row > 7500 or col < 2200 or col > 2600:
    #             continue
            img = cv2.imread(os.path.join(data_dir,name))
            if img.shape != (256,256,3):
                del name
                continue
            img = cv2.imread(os.path.join(data_dir,name))
    #         img_gamma = adjust_gamma(img, gamma)
            x_test.append(img)
            x_name_list.append(name)

        if len(x_test) == 0:
            continue

        x_test_np = np.array(x_test)
        x_test_np = x_test_np.astype(np.float64)
        x_test_np  = x_test_np / (win_size-1)
        x_test_np  = (x_test_np  - img_mean) / img_std

        x_torch = torch.from_numpy(x_test_np.copy()).type(torch.FloatTensor).to(device)
        x_torch = torch.moveaxis(x_torch, 3, 1)

        ##### prediction
        node_cat_pred_np, node_reg_pred_np, adj_pred_np, pred_dec_inputs_np, pred_dec_mask_np,  \
           enc_attn_np, dec_attn_np = predict(model, x_torch)

        for img_idx in range(node_cat_pred_np.shape[0]):
            ####### get predicted nodes
            pred = np.argmax(node_cat_pred_np[img_idx,:,:], axis=-1)
            pred_node = node_reg_pred_np[img_idx,:,:]

            pos_idx = np.where(pred!=0)[0]
            if pos_idx.shape[0] < num_nodes_thres:
                continue

            row, col = x_name_list[img_idx][:-4].split('_')
            row, col = int(row), int(col)
            ####### get predicted edges
            num_nodes = np.where(pred_dec_mask_np[img_idx]==0)[0].shape[0]
            adj_where = np.argsort(adj_pred_np[img_idx,:num_nodes,:num_nodes], axis=1)[:, -2:]

            for i, ind in enumerate(adj_where):
                x1, y1 = (pred_dec_inputs_np[img_idx, i]*(win_size-1)).astype('int32')
                for e in ind:
                    x2, y2 = (pred_dec_inputs_np[img_idx, e]*(win_size-1)).astype('int32')
                    if (x1-x2)**2 + (y1-y2)**2 > 40**2:
                        continue
                    x1_in_map, y1_in_map = x1+row, y1+col
                    x2_in_map, y2_in_map = x2+row, y2+col
                    node1_in_dict, node2_in_dict  = False, False
                    s_ind, e_ind = None, None
                    for x_buff in range(-buffer, buffer, 1):
                        for y_buff in range(-buffer, buffer, 1):
                            if [x1_in_map+x_buff, y1_in_map+y_buff] in nodes_dict.values():
                                new_x1 = x1_in_map+x_buff//2
                                new_y1 = y1_in_map+y_buff//2
                                s_ind = list(nodes_dict.keys())[list(nodes_dict.values()).index([x1_in_map+x_buff, y1_in_map+y_buff])]
                                nodes_dict[s_ind] = [new_x1, new_y1]
                                node1_in_dict = True
                                break
                        if node1_in_dict:
                            break
                    for x_buff in range(-buffer, buffer, 1):
                        for y_buff in range(-buffer, buffer, 1):
                            if [x2_in_map+x_buff, y2_in_map+y_buff] in nodes_dict.values():
                                new_x2, new_y2 = x2_in_map+x_buff//2, y2_in_map+y_buff//2
                                e_ind = list(nodes_dict.keys())[list(nodes_dict.values()).index([x2_in_map+x_buff, y2_in_map+y_buff])]
                                nodes_dict[e_ind] = [new_x2, new_y2]
                                node2_in_dict = True
                                break
                        if node2_in_dict:
                            break
                    if not node1_in_dict:
                        s_ind = len(nodes_dict.keys())
                        nodes_dict[s_ind] = [x1_in_map, y1_in_map]
                    if not node2_in_dict:
                        e_ind = len(nodes_dict.keys())
                        nodes_dict[e_ind] = [x2_in_map, y2_in_map]

                    if (s_ind, e_ind) in edges_dict: 
                        edges_dict[(s_ind, e_ind)] += 1
                    else:
                        edges_dict[(s_ind, e_ind)] = 1
                    if(e_ind, s_ind) in edges_dict:
                        edges_dict[(e_ind, s_ind)] += 1
                    else:
                        edges_dict[(e_ind, s_ind)] = 1
    return nodes_dict, edges_dict