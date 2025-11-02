import torch
from torch_geometric.data import Data, Dataset
# from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import numpy as np
from copy import deepcopy

graphs = []
file_pfx = ""
path_pfx = "./data/"

dataset_type = "2Dele"


def get_x_y_from_field_npy(is_bdry_node, is_bdry_face):
    if dataset_type == "2Dele":
        sample_amt =  torch.from_numpy(np.load(path_pfx+"E_primal.npy")).shape[0]
        E_p,E_d = torch.from_numpy(np.load(path_pfx+"E_primal.npy")).view(sample_amt,-1,2), torch.from_numpy(np.load(path_pfx+"E_dual.npy")).view(sample_amt,-1,2)
        D_p,D_d = torch.from_numpy(np.load(path_pfx+"D_primal.npy")).view(sample_amt,-1,2), torch.from_numpy(np.load(path_pfx+"D_dual.npy")).view(sample_amt,-1,2)
        rho_p,rho_d = torch.from_numpy(np.load(path_pfx+"rho_primal.npy")).view(sample_amt,-1,1), torch.from_numpy(np.load(path_pfx+"rho_dual.npy")).view(sample_amt,-1,1)
    elif dataset_type == "2Dmag":
        sample_amt =  torch.from_numpy(np.load(path_pfx+"H_primal.npy")).shape[0]
        E_p,E_d = torch.from_numpy(np.load(path_pfx+"H_primal.npy")).view(sample_amt,-1,2), torch.from_numpy(np.load(path_pfx+"H_dual.npy")).view(sample_amt,-1,2)
        D_p,D_d = torch.from_numpy(np.load(path_pfx+"B_primal.npy")).view(sample_amt,-1,2), torch.from_numpy(np.load(path_pfx+"B_dual.npy")).view(sample_amt,-1,2)
        rho_p,rho_d = torch.from_numpy(np.load(path_pfx+"J_primal.npy")).view(sample_amt,-1,1), torch.from_numpy(np.load(path_pfx+"J_dual.npy")).view(sample_amt,-1,1)
    else:
        assert False

    def fn(x): #field normalization  
        norm = torch.norm(x, dim=-1).view(-1,1)
        mean, std = torch.mean(norm), torch.std(norm)
        return x/mean

    sn = lambda x: (x-torch.mean(x))/torch.std(x) #scalar normalization

    E_p,E_d,D_p,D_d,rho_p,rho_d= fn(E_p), fn(E_d), fn(D_p), fn(D_d), sn(rho_p), sn(rho_d)

    primal_fields, dual_fields = torch.cat((E_p,D_p,rho_p),dim=-1),  torch.cat((E_d,D_d,rho_d),dim=-1)

    primal_x_lst, dual_x_lst,primal_y_lst,dual_y_lst= [],[],[],[]

    for t in range(sample_amt):
        observed_primal_bdry = torch.where(is_bdry_node.expand(-1,4), primal_fields[t,:,:4], torch.zeros_like(primal_fields[t,:,:4])).view(-1,4)
        observed_primal_src = primal_fields[t,:,4:]
        primal_x_lst.append(torch.cat([observed_primal_bdry, observed_primal_src],dim=-1).to(torch.float32))

        observed_dual_bdry = torch.where(is_bdry_face.expand(-1,4), dual_fields[t,:,:4], torch.zeros_like(dual_fields[t,:,:4])).view(-1,4)
        observed_dual_src = dual_fields[t,:,4:]
        dual_x_lst.append(torch.cat([observed_dual_bdry, observed_dual_src],dim=-1).to(torch.float32))
        
        primal_y_lst.append(primal_fields[t,:, :4].to(torch.float32))
        dual_y_lst.append(dual_fields[t,:, :4].to(torch.float32))
    return primal_x_lst, dual_x_lst, primal_y_lst, dual_y_lst

def get_adj_from_pts():
    Btsr = lambda x : torch.tensor(x, dtype=torch.int64)

    nnu,eel,eeu,ffl,edge_idx,tri_idx,vedge_idx = torch.load(path_pfx+file_pfx+"adj_maps_nnu_eelu_ffl.pt")
    ef,ne = torch.load(path_pfx+file_pfx+"cobdry_maps_ef_ne.pt")  
    nnu,eel,eeu,ffl,edge_idx,tri_idx,vedge_idx  = Btsr(nnu),Btsr(eel),Btsr(eeu),Btsr(ffl),Btsr(edge_idx),Btsr(tri_idx) ,Btsr(vedge_idx) 
    ef, ne =Btsr(ef),Btsr(ne)
    ef[:, [0, 1]] = ef[:, [1, 0]]
    ne[:, [0, 1]] = ne[:, [1, 0]]

    return [nnu,eel,eeu,ffl,ef,ne,edge_idx,tri_idx, vedge_idx]

def get_vol_from_pts():
    pev,pfv,vev,vfv= [x.to(torch.float32) for x in torch.load(path_pfx+file_pfx+"primal_dual_edge_face_vols.pt")]
    return [pev,pfv,vev,vfv]

def get_bdry_markers_from_pts():
    is_bdry_node,  is_bdry_edge, is_bdry_face = torch.load(path_pfx+file_pfx+"is_nef_bdry.pt")
    return [is_bdry_node,  is_bdry_edge, is_bdry_face]

def get_dedge2pedge_toolkits():
   return None

class HOGraphDataset(Dataset):    
    def __init__(self, graphs=None, p_x_lst=None, d_x_lst=None, p_y_lst=None, d_y_lst=None,
                 adjs=None, vols_lst=None,
                 #volumes of each ele.
                 poses=None, whitneys=None,edge_tangents=None, dual_edge_tangents=None,
                 #bdry markers
                 bdry_markers=None, levels=None
                 #dedge2pedge toolkits
                #  query_len_ratio=None, OH_to_OA_matrix=None,AO_idx_in_cf_tb=None,pface_pos_in_pcell=None,padedd_dual_edge_bdry=None,
                #  proper_ctrs=None,pedge_ratio=None
                 ):
        super().__init__()
        if graphs == None:
            self.graphs = [
                Data(p_x=x0, d_x=x1, p_y=y0,d_y=y1,
                     vols=vols, bdry_markers=bdry,levels = lvl, 
                     poses=pos, adjs=adj,
                     edge_tangents = et, dual_edge_tangents = det,  whitneys=whitney,
                    #  query_len_ratio=qlr0, OH_to_OA_matrix=otom,AO_idx_in_cf_tb=aiict,pface_pos_in_pcell=ppip,padedd_dual_edge_bdry=pdeb,
                    #  proper_ctrs=pc,pedge_ratio=pr
                     )
                for x0,x1,y0,y1,adj,vols,pos,whitney,bdry,et,det, lvl
                in zip(p_x_lst,d_x_lst,p_y_lst,d_y_lst, 
                       adjs,vols_lst,poses, whitneys,bdry_markers,
                       edge_tangents, dual_edge_tangents, levels, 
                    #    query_len_ratio, OH_to_OA_matrix,AO_idx_in_cf_tb,pface_pos_in_pcell,padedd_dual_edge_bdry,proper_ctrs,pedge_ratio
                       )
            ]
        else:
            self.graphs = deepcopy(graphs)

    def len(self):
        return len(self.graphs)
    
    def get(self, idx):
        return self.graphs[idx]
    
    def get_partition_datasets(self,proportion=[0.8,0.1,0.1]):
        train_amt,valid_amt = int(proportion[0]*self.len()), int(proportion[1]*self.len())
        lengths = [train_amt, valid_amt, len(self.graphs) - train_amt - valid_amt]
        train, val, test =  random_split(self.graphs, lengths, generator=torch.Generator().manual_seed(42))

        return HOGraphDataset(graphs=train), HOGraphDataset(graphs=val),HOGraphDataset(graphs=test)

# create DataLoader
def merge(batch):
    n_ofs, e_ofs, f_ofs = 0,0,0
    
    ctr1_ofs, ctr2_ofs, ctr3_ofs = 0,0, 0
    dctr1_ofs, dctr2_ofs, dctr3_ofs = 0,0, 0

    merged_batch = Data()
    merged_batch.p_x = torch.cat([d.p_x for d in batch], dim=0)     # [batch_size, num_nodes, features]
    merged_batch.d_x = torch.cat([d.d_x for d in batch], dim=0)     # [batch_size, num_nodes, features]
    
    merged_batch.p_y = torch.cat([d.p_y for d in batch], dim=0)     # [batch_size, num_nodes, features]
    merged_batch.d_y = torch.cat([d.d_y for d in batch], dim=0)     # [batch_size, num_nodes, features]
    
    merged_batch.node_pos = torch.cat([d.poses[0] for d in batch], dim=0)
    merged_batch.vnode_pos = torch.cat([d.poses[1] for d in batch], dim=0)
    merged_batch.edge_pos = torch.cat([d.poses[2] for d in batch], dim=0)

    # merged_batch.edge_tangents = torch.cat([d.edge_tangents for d in batch], dim=0)
    # merged_batch.face_normals = torch.cat([d.face_normals for d in batch], dim=0)

    merged_batch.pedge_vol = torch.cat([d.vols[0] for d in batch], dim=0).view(-1,1)
    merged_batch.pface_vol = torch.cat([d.vols[1] for d in batch], dim=0).view(-1,1) 
    merged_batch.vedge_vol = torch.cat([d.vols[2] for d in batch], dim=0).view(-1,1) 
    merged_batch.vface_vol = torch.cat([d.vols[3] for d in batch], dim=0).view(-1,1) 

    merged_batch.pnode_vol = torch.ones_like(merged_batch.vface_vol, dtype=torch.float32).view(-1,1)
    merged_batch.vnode_vol = torch.ones_like(merged_batch.pface_vol, dtype=torch.float32).view(-1,1)

    merged_batch.edge_tangents = torch.cat([d.edge_tangents for d in batch], dim=0).view(-1,2) 
    merged_batch.dual_edge_tangents = torch.cat([d.dual_edge_tangents for d in batch], dim=0).view(-1,2) 

    merged_batch.edge_whitney_vec = torch.cat([d.whitneys[1] for d in batch], dim=0) 
    merged_batch.dedge_whitney_vec = torch.cat([d.whitneys[3] for d in batch], dim=0) 

    merged_batch.is_bdry_node =  torch.cat([d.bdry_markers[0] for d in batch], dim=0)
    merged_batch.is_bdry_edge =  torch.cat([d.bdry_markers[1] for d in batch], dim=0)
    merged_batch.is_bdry_face =  torch.cat([d.bdry_markers[2] for d in batch], dim=0)
    
    node_node_upper_lst, edge_edge_lower_lst, edge_edge_upper_lst,face_face_lower_lst =[], [], [], []
    face_edge_lst, edge_node_lst = [], []
    edge_idx_lst, face_idx_lst, vedge_idx_lst = [], [], []

    edge_whitney_idx_lst, dedge_whitney_idx_lst = [], []

    aggr_edge2ctr1_idx_lst,aggr_ctr12ctr2_idx_lst,aggr_ctr22ctr3_idx_lst = [], [], []
    edge_srch_from_ctr1_lst, edge_srch_from_ctr2_lst,edge_srch_from_ctr3_lst =  [], [], []
    aggr_d_edge2ctr1_idx_lst,aggr_d_ctr12ctr2_idx_lst,aggr_d_ctr22ctr3_idx_lst = [], [], []
    edge_d_srch_from_ctr1_lst, edge_d_srch_from_ctr2_lst,edge_d_srch_from_ctr3_lst =  [], [], []

    for d in batch:
        edge_to_ctr_lst,aggr_edge_to_ctr_idx_lst, edge_to_higher_ctrs_lst, edge_srch_idx_lst \
             ,dedge_to_ctr_lst,aggr_dedge_to_ctr_idx_lst, dedge_to_higher_ctrs_lst, dedge_srch_idx_lst = d.levels

        n_amt, e_amt, f_amt = d.p_x.shape[0], d.vols[0].shape[0], d.vols[1].shape[0]        
        ctr1_amt, ctr2_amt, ctr3_amt = edge_to_ctr_lst[0].max()+1, edge_to_ctr_lst[1].max()+1,edge_to_ctr_lst[2].max()+1
        dctr1_amt, dctr2_amt, dctr3_amt = dedge_to_ctr_lst[0].max()+1, dedge_to_ctr_lst[1].max()+1,dedge_to_ctr_lst[2].max()+1

        node_node_upper_lst.append(d.adjs[0].add(torch.tensor([n_ofs, n_ofs, e_ofs, 0])))
        edge_edge_lower_lst.append(d.adjs[1].add(torch.tensor([e_ofs, e_ofs, n_ofs, 0])))
        edge_edge_upper_lst.append(d.adjs[2].add(torch.tensor([e_ofs, e_ofs, f_ofs, 0])))
        face_face_lower_lst.append(d.adjs[3].add(torch.tensor([f_ofs, f_ofs, e_ofs, 0])))

        edge_node_lst.append(d.adjs[5].add(torch.tensor([e_ofs, n_ofs, 0])))
        face_edge_lst.append(d.adjs[4].add(torch.tensor([f_ofs, e_ofs, 0])))

        edge_idx_lst.append(d.adjs[6].add(torch.tensor([n_ofs, n_ofs])))
        face_idx_lst.append(d.adjs[7].add(torch.tensor([n_ofs, n_ofs, n_ofs])))
        vedge_idx_lst.append(d.adjs[8].add(torch.tensor([f_ofs, f_ofs])))
        
        # edge2ctr1_lst.append(edge_to_ctr_lst[0].add(torch.tensor([ctr1_ofs])))
        # ctr12ctr2_lst.append(edge_to_ctr_lst[1].add(torch.tensor([ctr2_ofs])))
        # ctr22ctr3_lst.append(edge_to_ctr_lst[2].add(torch.tensor([ctr3_ofs])))

        aggr_edge2ctr1_idx_lst.append(aggr_edge_to_ctr_idx_lst[0].add(torch.tensor([e_ofs, ctr1_ofs])))
        aggr_ctr12ctr2_idx_lst.append(aggr_edge_to_ctr_idx_lst[1].add(torch.tensor([ctr1_ofs, ctr2_ofs])))
        aggr_ctr22ctr3_idx_lst.append(aggr_edge_to_ctr_idx_lst[2].add(torch.tensor([ctr2_ofs, ctr3_ofs])))

        # edgeInctr1_lst.append(edge_to_higher_ctrs_lst[0].add(torch.tensor([ctr1_ofs]))) 
        # edgeInctr2_lst.append(edge_to_higher_ctrs_lst[1].add(torch.tensor([ctr2_ofs]))) 
        # edgeInctr3_lst.append(edge_to_higher_ctrs_lst[2].add(torch.tensor([ctr3_ofs]))) 

        edge_srch_from_ctr1_lst.append(edge_srch_idx_lst[0].add(torch.tensor([e_ofs, ctr1_ofs])))
        edge_srch_from_ctr2_lst.append(edge_srch_idx_lst[1].add(torch.tensor([e_ofs, ctr2_ofs])))
        edge_srch_from_ctr3_lst.append(edge_srch_idx_lst[2].add(torch.tensor([e_ofs, ctr3_ofs]))) 

        aggr_d_edge2ctr1_idx_lst.append(aggr_dedge_to_ctr_idx_lst[0].add(torch.tensor([e_ofs, dctr1_ofs])))
        aggr_d_ctr12ctr2_idx_lst.append(aggr_dedge_to_ctr_idx_lst[1].add(torch.tensor([dctr1_ofs, dctr2_ofs])))
        aggr_d_ctr22ctr3_idx_lst.append(aggr_dedge_to_ctr_idx_lst[2].add(torch.tensor([dctr2_ofs, dctr3_ofs])))
        edge_d_srch_from_ctr1_lst.append(dedge_srch_idx_lst[0].add(torch.tensor([e_ofs, dctr1_ofs])))
        edge_d_srch_from_ctr2_lst.append(dedge_srch_idx_lst[1].add(torch.tensor([e_ofs, dctr2_ofs])))
        edge_d_srch_from_ctr3_lst.append(dedge_srch_idx_lst[2].add(torch.tensor([e_ofs, dctr3_ofs]))) 
        
        edge_whitney_idx_lst.append(d.whitneys[0].add(torch.tensor([e_ofs, n_ofs])))
        dedge_whitney_idx_lst.append(d.whitneys[2].add(torch.tensor([e_ofs, n_ofs])))

        n_ofs, e_ofs, f_ofs = n_ofs+n_amt, e_ofs+e_amt, f_ofs+f_amt
        ctr1_ofs, ctr2_ofs, ctr3_ofs =  ctr1_ofs+ctr1_amt, ctr2_ofs+ctr2_amt,  ctr3_ofs+ctr3_amt
        dctr1_ofs, dctr2_ofs, dctr3_ofs =  dctr1_ofs+dctr1_amt, dctr2_ofs+dctr2_amt,  dctr3_ofs+dctr3_amt

        # node_ctr_map_lst.append(node_ctr_lst.add(torch.tensor([n_ofs])))
        # ctr_ctr2_map_lst.append(ctr_ctr2_lst.add(torch.tensor([ctr_ofs])))
        # ctr_node_idx_lst.append(ctr_node_idx.add(torch.tensor([ctr_ofs, n_ofs])))
        # ctr2_ctr_idx_lst.append(ctr2_ctr_idx.add(torch.tensor([ctr2_ofs, ctr_ofs])))
        # ctr_face_idx_lst.append(ctr_face_idx.add(torch.tensor([ctr_ofs, f_ofs])))

        # node_in_ctr_map_lst.append(node_ctr_map.add(torch.tensor([ctr_ofs])))
        # node_in_ctr2_map_lst.append(node_ctr2_map.add(torch.tensor([ctr2_ofs])))
        # node_ctr_srch_idx_lst.append(node_ctr_srch_idx.add(torch.tensor([n_ofs, ctr_ofs])))
        # node_ctr2_srch_idx_lst.append(node_ctr2_srch_idx.add(torch.tensor([n_ofs, ctr2_ofs])))

        #torch.save([edge_whitney_idx,edge_whitney_vec,dedge_whitney_idx,dedge_whitney_vec],  file_name+'whitney_eles.pt')
        # padedd_dual_edge_bdry_lst.append(d.padedd_dual_edge_bdry.add(torch.tensor([c_ofs, c_ofs])))
        # proper_ctrs_lst.append(d.proper_ctrs.add(torch.tensor([c_ofs, c_ofs, c_ofs])))

    merged_batch.node_node_upper_idx = torch.cat(node_node_upper_lst, dim=0)
    merged_batch.edge_edge_lower_idx = torch.cat(edge_edge_lower_lst, dim=0)
    merged_batch.edge_edge_upper_idx = torch.cat(edge_edge_upper_lst, dim=0)
    merged_batch.face_face_lower_idx = torch.cat(face_face_lower_lst, dim=0)

    merged_batch.edge_node_idx = torch.cat(edge_node_lst, dim=0)
    merged_batch.face_edge_idx = torch.cat(face_edge_lst, dim=0)

    merged_batch.edge_idx = torch.cat(edge_idx_lst, dim=0)
    merged_batch.face_idx = torch.cat(face_idx_lst, dim=0)
    merged_batch.vedge_idx = torch.cat(vedge_idx_lst, dim=0)

    # merged_batch.edge2ctr1 = torch.cat(edge2ctr1_lst,dim=0)
    # merged_batch.ctr12ctr2 = torch.cat(ctr12ctr2_lst,dim=0)
    # merged_batch.ctr22ctr3 = torch.cat(ctr22ctr3_lst,dim=0)

    merged_batch.aggr_edge2ctr1_idx = torch.cat(aggr_edge2ctr1_idx_lst,dim=0)
    merged_batch.aggr_ctr12ctr2_idx = torch.cat(aggr_ctr12ctr2_idx_lst,dim=0)
    merged_batch.aggr_ctr22ctr3_idx = torch.cat(aggr_ctr22ctr3_idx_lst,dim=0)

    # merged_batch.edgeInctr1 = torch.cat(edgeInctr1_lst,dim=0) 
    # merged_batch.edgeInctr2 = torch.cat(edgeInctr2_lst,dim=0) 
    # merged_batch.edgeInctr3 = torch.cat(edgeInctr3_lst,dim=0) 

    merged_batch.edge_srch_from_ctr1 = torch.cat(edge_srch_from_ctr1_lst,dim=0)
    merged_batch.edge_srch_from_ctr2 = torch.cat(edge_srch_from_ctr2_lst,dim=0)
    merged_batch.edge_srch_from_ctr3 = torch.cat(edge_srch_from_ctr3_lst,dim=0)

    merged_batch.aggr_d_edge2ctr1_idx = torch.cat(aggr_d_edge2ctr1_idx_lst,dim=0)
    merged_batch.aggr_d_ctr12ctr2_idx = torch.cat(aggr_d_ctr12ctr2_idx_lst,dim=0)
    merged_batch.aggr_d_ctr22ctr3_idx = torch.cat(aggr_d_ctr22ctr3_idx_lst,dim=0)

    merged_batch.edge_d_srch_from_ctr1 = torch.cat(edge_d_srch_from_ctr1_lst,dim=0)
    merged_batch.edge_d_srch_from_ctr2 = torch.cat(edge_d_srch_from_ctr2_lst,dim=0)
    merged_batch.edge_d_srch_from_ctr3 = torch.cat(edge_d_srch_from_ctr3_lst,dim=0)

    # merged_batch.node_ctr_map = torch.cat(node_ctr_map_lst,dim=0)
    # merged_batch.ctr_ctr2_map = torch.cat(ctr_ctr2_map_lst,dim=0)
    # merged_batch.ctr_node_idx = torch.cat(ctr_node_idx_lst,dim=0)
    # merged_batch.ctr2_ctr_idx = torch.cat(ctr2_ctr_idx_lst,dim=0)
    # merged_batch.ctr_face_idx = torch.cat(ctr_face_idx_lst,dim=0)

    # merged_batch.node_in_ctr_map = torch.cat(node_in_ctr_map_lst,dim=0)
    # merged_batch.node_in_ctr2_map = torch.cat(node_in_ctr2_map_lst,dim=0)
    # merged_batch.node_ctr_srch_idx = torch.cat(node_ctr_srch_idx_lst, dim=0)
    # merged_batch.node_ctr2_srch_idx = torch.cat(node_ctr2_srch_idx_lst, dim=0)

    merged_batch.edge_whitney_idx = torch.cat(edge_whitney_idx_lst, dim=0)
    merged_batch.dedge_whitney_idx = torch.cat(dedge_whitney_idx_lst, dim=0)

    # merged_batch.padedd_dual_edge_bdry = torch.cat(padedd_dual_edge_bdry_lst, dim=0)
    # merged_batch.proper_ctrs = torch.cat(proper_ctrs_lst, dim=0)

    merged_batch.cur_bz = len(batch)

    return merged_batch

def get_loader(dataset, bz):
    return DataLoader(
        dataset,
        batch_size=bz,
        shuffle=False,
        collate_fn=merge,  
        num_workers=4,
        pin_memory=True
    )

if __name__ == '__main__':
    adjs = get_adj_from_pts()
    vols = get_vol_from_pts()
    
    bdry_markers = get_bdry_markers_from_pts()

    primal_x_lst, dual_x_lst,  primal_y_lst, dual_y_lst= get_x_y_from_field_npy(bdry_markers[0], bdry_markers[2])

    poses = torch.load(path_pfx+file_pfx+"node_vnode_pos.pt")
    whitneys = torch.load(path_pfx+file_pfx+"whitney_eles.pt")

    edge_tangents, dual_edge_tangents = torch.load(path_pfx+file_pfx+"edge_dualedge_tangents.pt")
    levels = torch.load(path_pfx+file_pfx+"level.pt")

    num_graphs,num_x_feats, num_y_feats = len(primal_x_lst), primal_x_lst[0].shape[-1], primal_y_lst[0].shape[-1]
    N = num_graphs

    train_set, val_set, test_set = HOGraphDataset(
        p_x_lst=primal_x_lst,d_x_lst=dual_x_lst, p_y_lst=primal_y_lst,d_y_lst=dual_y_lst, 
        adjs=[adjs]*N, vols_lst=[vols]*N, poses = [poses]*N, whitneys=[whitneys]*N,
        bdry_markers= [bdry_markers]*N, 
        edge_tangents=[edge_tangents]*N, dual_edge_tangents=[dual_edge_tangents]*N,
        levels = [levels]*N
        ).get_partition_datasets([0.8,0.1,0.1])

    torch.save(train_set, path_pfx+file_pfx+'train_set2D.pth')
    torch.save(test_set, path_pfx+file_pfx+'test_set2D.pth')
    torch.save(val_set, path_pfx+file_pfx+'val_set2D.pth')
    print("Datasets Saved.")

    train_loader  = get_loader(train_set, bz=1)
    print(train_set[0])
    batch = next(iter(train_loader))
    print(batch)


