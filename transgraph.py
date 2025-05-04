import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import _get_clones

from torch_geometric.nn.conv import TransformerConv

class TRANSGRAPH(nn.Module):
    def __init__(self, in_channels, out_channels, num_relations, heads, num_layers):
        super(TRANSGRAPH, self).__init__()
        assert in_channels == heads * out_channels, 'TRANSGRAPH in_channels must equal heads * out_channels'

        self.num_layers = num_layers
        
        conv = TransformerConv(in_channels = in_channels, 
                               out_channels = out_channels,
                               heads = heads,
                               beta = False,
                               edge_dim = None,
                               )
        self.convs = _get_clones(conv, num_layers)

        self.fflayer1 = nn.Linear(in_channels, 2*in_channels) 
        self.fflayer2 = nn.Linear(2*in_channels, in_channels)

        self.LayerNorm = nn.LayerNorm(normalized_shape = 512)
        

    def forward(self, x, edge_index, edge_type,edge_repre):
        enc_emb = x

        for i in range(self.num_layers):
            x = self.convs[i](x=x, edge_index=edge_index)
            x = x + enc_emb
          
        # x = x + enc_emb    # skip connection
        x = self.LayerNorm (x)
        x = self.fflayer2(F.relu(self.fflayer1(x))) + x
        x = self.LayerNorm(x)

        return  x  + enc_emb



    