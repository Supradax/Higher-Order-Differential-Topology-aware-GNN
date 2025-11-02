import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn import GCNConv, GATConv, TransformerConv
import basics

class LowerAdjTransformer(nn.Module):
    """Graph transformer for lower adjacent with multiple layers and residual connections."""
    def __init__(self, input_dim, output_dim, geo_dim, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.edge_encoder = nn.Linear(input_dim*2 + geo_dim + 1, output_dim) 
        
        self.transformer_convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else output_dim
            self.transformer_convs.append(
                TransformerConv(
                    in_channels=in_channels,
                    out_channels=output_dim,
                    edge_dim=output_dim,
                    heads=3,
                    concat=False
                )
            )
        self.residual_linears = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.residual_linears.append(nn.Linear(input_dim, output_dim))
            else:
                self.residual_linears.append(nn.Identity()) 
        self.ctr_lin = nn.Linear(input_dim + output_dim, output_dim)

    def forward(self, h, lower_adj_feat, lower_adj):
        src_nodes = lower_adj[:, 0]
        dst_nodes = lower_adj[:, 1]
        direction_weights = lower_adj[:, 3:4]  
        
        edge_features = torch.cat([
            h[dst_nodes], 
            h[src_nodes] * direction_weights,
            direction_weights,  
            lower_adj_feat
        ], dim=-1)
        
        edge_attr = self.edge_encoder(edge_features)
        edge_index = torch.stack([src_nodes, dst_nodes], dim=0)
        
        h_current = h
        for i in range(self.num_layers):
            residual = self.residual_linears[i](h_current)
            h_transformed = self.transformer_convs[i](h_current, edge_index, edge_attr=edge_attr)
            h_current = F.relu(h_transformed + residual)
        
        h_aggr = h_current
        return self.ctr_lin(torch.cat([h, h_aggr], dim=-1))

class PrimalCobdryTransformer(nn.Module):
    """Graph transformer for coboundry in primal graph."""
    def __init__(self, bdry_dim, cobdry_dim, output_dim):
        super().__init__()
        self.aggr_lin = nn.Linear(bdry_dim, output_dim)
        self.ctr_lin = nn.Linear(cobdry_dim, output_dim)
        self.face_transformers = nn.ModuleList([
            TransformerConv(
                output_dim,
                output_dim,
                edge_dim=bdry_dim,
                heads=3,
                concat=False
            ) for _ in range(3)
        ])
        self.edge_transformer = TransformerConv(
            output_dim,
            bdry_dim,
            edge_dim=bdry_dim, 
            heads=3,
            concat=False
        )

    def forward(self, h_E, h_F, data):
        cobdry_idx = data.face_edge_idx 
        h_aggr = scatter(
            src=h_E[cobdry_idx[:,1]] * cobdry_idx[:,2:],
            index=cobdry_idx[:,0],
            dim=0,
            reduce='add'
        )
        h_F = self.aggr_lin(h_aggr) + self.ctr_lin(h_F)
        face_face_index = data.face_face_lower_idx[:,0:2].t()
        for transformer in self.face_transformers:
            h_F_res = h_F  
            h_F = transformer(h_F,face_face_index,edge_attr=h_E[data.face_face_lower_idx[:,2]])
            h_F = h_F + h_F_res  

        edge_index = cobdry_idx[:,[0,1]].t()  
        edge_attr = h_F[cobdry_idx[:,0]] * cobdry_idx[:,2:] 
        
        edge_amt = h_E.shape[0]
        h_E = torch.cat([h_E, h_F], dim=0) 
        h_E = self.edge_transformer(
            h_E,
            edge_index.add(torch.tensor([edge_amt, 0], device=edge_index.device).long().view(2,1)),
            edge_attr=edge_attr 
        )[:edge_amt]
        return h_E

class DualCobdryTransformer(nn.Module):
    """Graph transformer for coboundry in dual graph."""
    def __init__(self, bdry_dim, cobdry_dim, output_dim):
        super().__init__()
        self.aggr_lin = nn.Linear(bdry_dim, output_dim)
        self.ctr_lin = nn.Linear(cobdry_dim, output_dim)
        

        self.edge_transformer = TransformerConv(
            output_dim,
            bdry_dim,
            edge_dim=bdry_dim,  
            heads=3,
            concat=False
        )
        self.face_transformers = nn.ModuleList([
            TransformerConv(
                output_dim,
                output_dim,
                edge_dim=bdry_dim,
                heads=3,
                concat=False
            ) for _ in range(3)
        ])

    def forward(self, h_dE, h_dF, data):
        cobdry_idx = data.edge_node_idx 
        
        h_aggr = scatter(
            src=h_dE[cobdry_idx[:,0]] * cobdry_idx[:,2:],
            index=cobdry_idx[:,1],
            dim=0,
            reduce='add'
        )
        h_dF = self.aggr_lin(h_aggr) + self.ctr_lin(h_dF)

        dface_dface_index = data.node_node_upper_idx[:,0:2].t()  
        for transformer in self.face_transformers:
            h_F_res = h_dF 
            h_dF = transformer(
                h_dF,
                dface_dface_index,
                edge_attr=h_dE[data.node_node_upper_idx[:,2]]  
            )
            h_dF = h_dF + h_F_res  

        edge_index = cobdry_idx[:,[1,0]].t()  
        edge_attr = h_dF[cobdry_idx[:,1]] * cobdry_idx[:,2:]  
        dE_amt = h_dE.shape[0]
        h_dE = self.edge_transformer(
            torch.cat([h_dE,h_dF]),
            edge_index.add(torch.tensor([dE_amt, 0],device=edge_index.device).long().view(2,1)),
            edge_attr=edge_attr  
        )[:dE_amt]
        return h_dE

class DEC_HOGNN(torch.nn.Module):
    """GNN solver for electro PDE"""
    def __init__(self):
        super().__init__()
        o_dim = 128
        # Vector field encoder
        self.E_encode_mlp = basics.MLP(8, o_dim, o_dim*2, LN=True)
        self.D_encode_mlp = basics.MLP(8, o_dim, o_dim*2, LN=True)
        
        # Geometric feature processing
        self.angle_pairs = [(0,0),(0,1),(1,0),(0,2),(1,1),(2,0),(0,3),(1,2),(2,1),(3,0)]
        self.len_factor = nn.Parameter(torch.ones(2, dtype=torch.float32))

        # Face encoder for primal and dual graph
        self.face_encode_mlp = basics.MLP(2+1+1+1, o_dim, o_dim, LN=True)
        self.dface_encode_mlp = basics.MLP(2+1+1+1, o_dim*2, o_dim, LN=True)

        # Message passing networks
        self.pe_lower_mpnn = LowerAdjTransformer(input_dim=o_dim, output_dim=o_dim, geo_dim=20)            # Primal Edge 
        self.de_lower_mpnn = LowerAdjTransformer(input_dim=o_dim, output_dim=o_dim, geo_dim=20)            # Dual Edge 
        
        self.pe_lower_mpnn2 = LowerAdjTransformer(input_dim=o_dim+8+o_dim, output_dim=64, geo_dim=20)
        self.de_lower_mpnn2 = LowerAdjTransformer(input_dim=o_dim+8+o_dim, output_dim=64, geo_dim=20)
        
        self.pe_cobdry_mpnn = PrimalCobdryTransformer(bdry_dim=o_dim, cobdry_dim=o_dim, output_dim=o_dim)  # Primal Edge
        self.de_cobdry_mpnn = DualCobdryTransformer(bdry_dim=o_dim, cobdry_dim=o_dim, output_dim=o_dim)    # Dual Edge

        self.E_mlp = basics.MLP(o_dim+8, o_dim, 1, n_layers=3, LN=True)  #decode_pe_mlp_out, p_x/d_x
        self.D_mlp = basics.MLP(o_dim+8, o_dim, 1, n_layers=3, LN=True)  #node-wise:28, edge-wise:o_dim+8+o_dim

    def field_encoder(self, E, D, rho, data):
        """
        The nodal electric field E and the face-centered electric displacement field D 
        are projected onto mesh edges and dual edges 
        to generate edge-level tangential flux features.
        """
        # Electric field in primal graph
        E_on_edges = scatter(
            index=data.edge_node_idx[:,0:1],
            src=E[data.edge_node_idx[:,1]],
            reduce='mean',
            dim=0
        )
        # [num_edges, 2] -> [num_edges, 1, 2]
        E_on_edges = E_on_edges.view(-1, E_on_edges.shape[-1]//2, 2)
        # Tangential flux density of the electric field at the edge
        # [num_edges, 1, 2], [num_edges, 2] -> [num_edges, 1]
        h_E = torch.einsum('ikj,ij->ik', E_on_edges.float(), data.edge_tangents.float())

        # Electric displacement field in dual graph
        D_on_dual_edges = scatter(
            index=data.face_edge_idx[:,1:2],
            src=D[data.face_edge_idx[:,0]],
            reduce='mean',
            dim=0
        )
        # [num_edges, 2] -> [num_edges, 1, 2]
        D_on_dual_edges = D_on_dual_edges.view(-1, D_on_dual_edges.shape[-1]//2, 2)
        # The flux density of the electric displacement field in the direction of the dual edge
        # [num_edges, 1, 2], [num_edges, 2] -> [num_edges, 1]
        h_dE = torch.einsum('ikj,ij->ik', D_on_dual_edges.float(), data.dual_edge_tangents.float())

        # Integral
        h_E, h_dE = h_E * data.pedge_vol, h_dE * data.vedge_vol
        return h_E, h_dE

    def lower_adj_encoder(self, data):
        """
        Encode the adjacency relationship between the original and dual edges 
        to generate multi-scale geometric features.
        """
        def angle_encoder(theta0, theta1):
            """-> [num_edges, 2*num_pairs]"""
            cos = torch.stack([torch.cos(2**m*theta0 + 2**n*theta1) for m,n in self.angle_pairs], dim=-1)
            sin = torch.stack([torch.sin(2**m*theta0 + 2**n*theta1) for m,n in self.angle_pairs], dim=-1)
            return torch.cat([cos, sin], dim=-1)
        
        # PRIMAL
        lower_adj, edge_tangents = data.edge_edge_lower_idx, data.edge_tangents
        
        # calculate the length of adjacent edges, exponential decay
        edge_len0, edge_len1 = data.pedge_vol[lower_adj[:, 0]],data.pedge_vol[lower_adj[:, 1]]
        edge_len0, edge_len1 = torch.exp(-self.len_factor[0]*edge_len0), torch.exp(-self.len_factor[1]*edge_len1)

        # calculate the angle between adjacent edges
        edge_dir0, edge_dir1 = edge_tangents[lower_adj[:,0]], edge_tangents[lower_adj[:,1]]
        edge_angle0, edge_angle1 = torch.atan2(edge_dir0[:,0], edge_dir0[:,1]), torch.atan2(edge_dir1[:,0], edge_dir1[:,1])
        
        # angle encoding feature, combine with length weight
        encoded_angles = angle_encoder(edge_angle0, edge_angle1)
        p_lower_adj_feat = encoded_angles * edge_len0 * edge_len1

        # DUAL
        lower_adj, dual_edge_tangents = data.edge_edge_upper_idx, data.dual_edge_tangents
        
        edge_len0, edge_len1 = data.vedge_vol[lower_adj[:, 0]],data.vedge_vol[lower_adj[:, 1]]

        edge_dir0, edge_dir1 = dual_edge_tangents[lower_adj[:,0]], dual_edge_tangents[lower_adj[:,1]]
        edge_angle0, edge_angle1 = torch.atan2(edge_dir0[:,0], edge_dir0[:,1]), torch.atan2(edge_dir1[:,0], edge_dir1[:,1])
        
        encoded_angles = angle_encoder(edge_angle0, edge_angle1)
        d_lower_adj_feat = encoded_angles * edge_len0 * edge_len1

        return p_lower_adj_feat.float(), d_lower_adj_feat.float()

    def face_encoder(self, data):
        """
        Encodes features of both 
        the original mesh face (the triangle elements) and 
        the dual mesh face (the area around the vertices).
        """
        # PRIMAL
        # charge density, node position in primal graph, face volume in dual graph, boundry face or not.
        rho, vnode_pos, face_vol, bdry_mask = data.d_x[:,4:].float(), data.vnode_pos.float(), data.pface_vol.float(), data.is_bdry_face.float()
        face_feat = torch.cat([rho, vnode_pos, face_vol, bdry_mask], dim=-1)
        face_feat = self.face_encode_mlp(face_feat)

        # DUAL
        # charge density, node position in dual graph, face volume in primal graph, boundry face or not.
        rho, node_pos, dface_vol, bdry_mask = data.p_x[:,4:].float(), data.node_pos.float(), data.vface_vol.float(), data.is_bdry_node.float()
        dface_feat = torch.cat([rho, node_pos, dface_vol, bdry_mask], dim=-1)
        dface_feat = self.dface_encode_mlp(dface_feat)
        
        return face_feat * face_vol, dface_feat * dface_vol

    def forward(self, data, integrals=False):
        E, D, rho = data.p_x[:, 0:2].float(), data.d_x[:,2:4].float(), data.d_x[:,4:].float()
        
        E = self.E_encode_mlp(torch.cat([data.node_pos.float(), data.p_x, data.is_bdry_node.float()], dim=-1))
        D = self.D_encode_mlp(torch.cat([data.vnode_pos.float(), data.d_x, data.is_bdry_face.float()], dim=-1))
        
        h_E, h_dE = self.field_encoder(E, D, rho, data)
        p_lower_adj_feat, d_lower_adj_feat = self.lower_adj_encoder(data)
        h_F, h_dF = self.face_encoder(data)

        h_E = self.pe_lower_mpnn(h_E,p_lower_adj_feat,data.edge_edge_lower_idx)
        h_dE = self.de_lower_mpnn(h_dE, d_lower_adj_feat,data.edge_edge_upper_idx)

        h_E = self.pe_cobdry_mpnn(h_E, h_F, data)
        h_dE = self.de_cobdry_mpnn(h_dE, h_dF, data)
        
        edge_idx, w_pidx, w_pvec = data.edge_idx, data.edge_whitney_idx, data.edge_whitney_vec
        p_x_on_edge = (data.p_x[[edge_idx[:, 0]]] + data.p_x[[edge_idx[:, 1]]]) / 2
        E_feats = torch.cat([
            data.edge_pos.float(), 
            p_x_on_edge, 
            data.is_bdry_edge.float()
        ], dim=-1)
        
        dedge_idx, w_didx, w_dvec = data.vedge_idx, data.edge_whitney_idx, data.edge_whitney_vec
        d_x_on_dedge = (data.d_x[[dedge_idx[:, 0]]] + data.d_x[[dedge_idx[:, 1]]] * data.is_bdry_edge.float()) / 2
        D_feats =  torch.cat([
            data.edge_pos.float(), 
            d_x_on_dedge, 
            data.is_bdry_edge.float()
        ], dim=-1)

        E, D = torch.cat([h_E, E_feats], dim=-1), torch.cat([h_dE, D_feats], dim=-1)

        E_int, D_int = self.E_mlp(E), self.D_mlp(D)
        
        E = scatter(index=w_pidx[:, 1:], src=E_int[w_pidx[:, 0]]*w_pvec, dim=0, reduce='mean')
        D = scatter(index=w_didx[:, 1:], src=D_int[w_didx[:, 0]]*w_dvec, dim=0, reduce='mean')

        return (E, D, E_int, D_int) if integrals else (E, D)
