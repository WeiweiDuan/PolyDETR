import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import sys
from util.misc import NestedTensor, nested_tensor_from_tensor_list
from .backbone import build_backbone
from .transformer_encoder import build_transformer as build_encoder
from .transformer_decoder import build_transformer as build_decoder
from .position_encoding import PositionEmbeddingSine_ as PositionEmbeddingSine
from .adj_layer import Attention_nosftm as predict_adj
from util.args import get_args_parser

parser = get_args_parser()
args = parser.parse_args(sys.argv[1:])

class graph(nn.Module):
    def __init__(self, backbone, encoder, decoder, num_classes=1, num_nodes=100, aux_loss=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.hidden_dim = encoder.d_model
        ##### map banckbone's output to low-dim
        self.input_proj = nn.Conv2d(backbone.num_channels, self.hidden_dim, kernel_size=1)
        self.backbone = backbone
        ##### map 2d coordinate to high-dim
        self.linear = nn.Linear(2, self.hidden_dim)
        ############################
        ##### encoder classifiers
        ##### classify grid into 0/1
        ############################
        self.node_classifier = MLP(self.hidden_dim, self.hidden_dim, 2, 3)
        ##### regression the precise location of node in each grid
        self.node_regression = MLP(self.hidden_dim, self.hidden_dim, 2, 3)
        ###########################
        ##### decoder adjacent classifier
        ############################
        self.adj_classifier = adj_classifier(self.hidden_dim) 
        self.aux_loss = aux_loss
        self.num_nodes = num_nodes
   
    def forward(self, enc_inputs: NestedTensor, dec_inputs: NestedTensor):
        backbone_in = enc_inputs.tensors
        if isinstance(backbone_in, (list, torch.Tensor)):
            backbone_in = nested_tensor_from_tensor_list(backbone_in)
        features, pos = self.backbone(backbone_in)
        src, mask = features[-1].decompose()
        
        assert mask is not None
        ##### encoder
        memory, enc_attn = self.encoder(self.input_proj(src), mask, pos[-1])
        enc_cat_outputs = self.node_classifier(memory)
        enc_reg_outputs = self.node_regression(memory)
        ##### decoder
        edge_inputs = dec_inputs.tensors
        dec_mask = dec_inputs.mask
        
        dec_input_embed = self.linear(edge_inputs)

        ##### add img feature to pos feature
        bs, num_nodes, f_len = dec_input_embed.shape[0], dec_input_embed.shape[1], memory.shape[-1]
        selected_memory = torch.zeros([bs, num_nodes, f_len], dtype=torch.float32).to(memory.device)
        num_grid_row = args.img_size//args.grid_size
        for b in range(bs):
            for i, m in enumerate(dec_mask[b]):
                if m == True:
                    continue
                x, y = int(edge_inputs[b, i][0]*(args.img_size-1)), int(edge_inputs[b, i][1]*(args.img_size-1))
                grid_x, grid_y = x//args.grid_size, y//args.grid_size
#                 print(x, y, grid_x, grid_y, memory[b, grid_x*16+grid_y].shape)
                selected_memory[b, i] = memory[b, grid_x*num_grid_row+grid_y]
        dec_fin = dec_input_embed + selected_memory
        dec_hs, dec_attn = self.decoder(dec_fin, dec_mask, pos[-1])
        
        adj_cls = self.adj_classifier(dec_hs, dec_mask)
        out = {'pred_cat_nodes': enc_cat_outputs, 'pred_reg_nodes': enc_reg_outputs,\
              'pred_adj': adj_cls, 'mask': dec_mask}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(dec_attn)
        return out, enc_attn, dec_attn
        
    @torch.jit.unused
    def _set_aux_loss(self, outputs):
        # as a dict having both a Tensor and a list.
        return [{'pred_conn': i} for i in outputs]
    
    def predict(self, enc_inputs):
        import numpy as np
        if isinstance(enc_inputs, (list, torch.Tensor)):
            backbone_in = nested_tensor_from_tensor_list(enc_inputs)
        features, pos = self.backbone(backbone_in)
        src, mask = features[-1].decompose()
        num_grid_row = args.img_size//args.grid_size
        assert mask is not None
        
        ###### encoder
        memory, enc_attn = self.encoder(self.input_proj(src), mask, pos[-1])
        enc_cat_outputs = self.node_classifier(memory)
        enc_reg_outputs = self.node_regression(memory)
        enc_cat_sm = torch.nn.functional.softmax(enc_cat_outputs, dim=-1)
        enc_cat_bin = (enc_cat_sm > 0.5)[:,:,1].type(torch.ByteTensor)
        enc_reg_outputs_sig = torch.nn.functional.sigmoid(enc_reg_outputs)

        ##### decoder
        bs = memory.shape[0]      
        dec_inputs = torch.ones((bs, self.num_nodes, 2)).to(memory.device)
        dec_mask = torch.ones((bs, self.num_nodes)).type(torch.BoolTensor).to(memory.device)
        n_grids = args.img_size // args.grid_size
        for i in range(bs):
            c = 0
            for j in range(num_grid_row**2):
                if enc_cat_bin[i,j] == 1:
                    grid_x, grid_y = j//n_grids, j%n_grids
                    x = enc_reg_outputs_sig[i, j, 0]*args.grid_size+args.grid_size*grid_x 
                    y = enc_reg_outputs_sig[i, j, 1]*args.grid_size+args.grid_size*grid_y
#                     print(enc_reg_outputs_sig[i, j], grid_x, grid_y, x, y)
#                     dec_inputs[i, c] = torch.from_numpy(np.array([x, y])).type(torch.FloatTensor).to(device)
                    dec_inputs[i, c] = torch.FloatTensor([x, y])
                    dec_mask[i, c] = 0
                    c += 1
                    if c == self.num_nodes:
                        break
        
        dec_inputs = dec_inputs/(args.img_size-1)
        
        dec_inputs_embed = self.linear(dec_inputs)
        
        ##### add img feature to pos feature
        bs, num_nodes, f_len = dec_inputs_embed.shape[0], dec_inputs_embed.shape[1], memory.shape[-1]
        selected_memory = torch.zeros([bs, num_nodes, f_len], dtype=torch.float32).to(memory.device)
        for b in range(bs):
            for i, m in enumerate(dec_mask[b]):
                if m == True:
                    continue
                x, y = int(dec_inputs[b, i][0]*(args.img_size-1)), int(dec_inputs[b, i][1]*(args.img_size-1))
                grid_x, grid_y = x//args.grid_size, y//args.grid_size
#                 print(x, y, grid_x, grid_y, memory[b, grid_x*16+grid_y].shape)
                selected_memory[b, i] = memory[b, grid_x*num_grid_row+grid_y]
        dec_fin = dec_inputs_embed + selected_memory
        dec_hs, dec_attn = self.decoder(dec_fin, dec_mask, pos[-1])
        adj_cls = self.adj_classifier(dec_hs, dec_mask)
        
        out = {'pred_cat_nodes': enc_cat_outputs, 'pred_reg_nodes': enc_reg_outputs,\
              'pred_adj': adj_cls, 'mask': dec_mask, 'dec_inputs': dec_inputs}
        return out, enc_attn, dec_attn                                
        

class SetCriterion(nn.Module):
    
    def __init__(self, num_classes, eos_coef, losses):
        super().__init__()
        self.num_classes = num_classes
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(2)
        empty_weight[0] = 5.0
        self.register_buffer('empty_weight', empty_weight)
    
    def loss_cat_nodes(self, outputs, targets):
        node_logits = outputs['pred_cat_nodes']
        node_logits = torch.nan_to_num(node_logits, nan=0.0, posinf=1.0)
       
        loss_ce = F.cross_entropy(node_logits.transpose(1, 2), targets['cat_nodes'])#, self.empty_weight)
        losses = {'loss_cat_nodes': loss_ce}
        return losses
    
    def loss_reg_nodes(self, outputs, targets):
        node_reg = outputs['pred_reg_nodes'].sigmoid()
        node_reg = torch.nan_to_num(node_reg, nan=0.0, posinf=1.0)
        loss_mse = F.mse_loss(node_reg, targets['reg_nodes'])
        reg_loss_mask = targets['cat_nodes']
        loss_ = torch.mean(loss_mse*reg_loss_mask)*self.eos_coef
        losses = {'loss_reg_nodes': loss_}
        return losses
    
    def loss_adj(self, outputs, targets):
        adj_logits = outputs['pred_adj']
        adj_logits = torch.nan_to_num(adj_logits, nan=0.0, posinf=1.0)
        mask = ~targets['mask']
        expand_mask = mask.unsqueeze(1).repeat(1, adj_logits.shape[-1], 1)
        for b in range(mask.shape[0]):
            num_nodes = torch.sum(mask[b]).item()
            expand_mask[b, num_nodes:, :] = 0
            expand_mask[b, :, num_nodes:] = 0

        loss = F.binary_cross_entropy_with_logits(adj_logits, targets['line'], reduction='none')
        loss = loss * expand_mask
        loss_ = torch.mean(loss)*1.0
        losses = {'loss_adj': loss_}
        return losses
    
    def loss_conn(self, outputs, targets):
        conn_logits = outputs['pred_conn'] 
        conn_logits = torch.nan_to_num(conn_logits, nan=0.0, posinf=1.0)
        mask = ~targets['mask']
        expand_mask = mask.unsqueeze(1).repeat(1, conn_logits.shape[-1], 1)
        for b in range(mask.shape[0]):
            num_nodes = torch.sum(mask[b]).item()
            expand_mask[b, num_nodes:, :] = 0
            expand_mask[b, :, num_nodes:] = 0
        loss = F.binary_cross_entropy(conn_logits, targets['conn'], reduction='none')
        loss = loss * expand_mask
        loss_ = torch.mean(loss)*1.0
        losses = {'loss_conn': loss_}
        return losses
    
    def get_loss(self, loss, outputs, targets, **kwargs):
        loss_map = {
            'loss_cat_nodes': self.loss_cat_nodes,
            'loss_reg_nodes': self.loss_reg_nodes,
            'loss_adj': self.loss_adj,
            'loss_conn': self.loss_conn
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, **kwargs)
    
    
    def forward(self, outputs, targets):
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            if loss == 'loss_conn':
                continue
            losses.update(self.get_loss(loss, outputs, targets))
        
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                for loss in ['loss_conn']:
                    l_dict = self.get_loss(loss, aux_outputs, targets)#, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
class adj_classifier(nn.Module):

    def __init__(self, dim):
        super(adj_classifier, self).__init__()
        self.cls_attn = predict_adj(dim)

    def forward(self, query, mask=None):
#         query = query.permute(1, 0, 2)
        adj_matrix = self.cls_attn(query, query)
        
        return adj_matrix
    
def build(args):
    num_classes = 1
    device = torch.device(args.device)
    
    backbone = build_backbone(args)
    encoder = build_encoder(args)
    decoder = build_decoder(args)
    sine_pos_enc = PositionEmbeddingSine()
    
    model = graph(
        backbone,
        encoder,
        decoder, 
        num_classes, 
        args.num_nodes
    )
    
    losses = ['loss_cat_nodes', 'loss_reg_nodes', 'loss_adj', 'loss_conn']
    criterion = SetCriterion(num_classes, 
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)

    return model, criterion