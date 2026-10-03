import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import rdchem

max_atom=50
node_feature_dim=10
edge_feature_dim=6
global_feature_dim=3
max_edges=128
def atom_feature(atom):
    chiral=atom.GetChiralTag()
    chrality=(
        1 if chiral==rdchem.ChiralType.CHI_TETRAHEDRAL_CW else
        -1 if chiral==rdchem.ChiralType.CHI_TETRAHEDRAL_CCW else
        0
    )
    return np.array([
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic()),
        atom.GetTotalNumHs(),
        int(atom.GetHybridization()==rdchem.HybridizationType.SP2),
        int(atom.GetHybridization()==rdchem.HybridizationType.SP3),
        int(atom.IsInRing()),
        atom.GetTotalValence(),
        chrality
    ],dtype=np.float32)

def bond_feautres(bond):
    bt=bond.GetBondType()
    return np.array([
        bt==rdchem.BondType.SINGLE,
        bt==rdchem.BondType.DOUBLE,
        bt==rdchem.BondType.TRIPLE,
        bond.GetIsAromatic(),
        bond.GetIsConjugated(),
        bond.IsInRing()
    ],dtype=np.float32)

class MolGraphPrep:
    def smilesTomol(self,smiles):
        mol=Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms()>max_atom:
            return None
        return mol
    def molTograph(self,mol):
        node_f=np.zeros((max_atom,node_feature_dim),np.float32)
        adj=np.zeros((max_atom,max_atom), np.float32)
        mask=np.zeros((max_atom,),np.float32)
        edge_f=np.zeros((max_atom,max_atom,edge_feature_dim),np.float32)
        edge_idx=[]

        for i,atom in enumerate(mol.GetAtoms()):
            node_f[i]=atom_feature(atom)
            mask[i]=1.0
        
        for bond in mol.GetBonds():
            i=bond.GetBeginAtomIdx()
            j=bond.GetEndAtomIdx()

            adj[i,j]=adj[j,i]=1.0
            bf=bond_feautres(bond)

            edge_f[i,j]=bf
            edge_f[j,i]=bf

            edge_idx.append([i,j])
            edge_idx.append([j,i])
        edge_idx=np.array(edge_idx,dtype=np.int32)

        return node_f,edge_idx,edge_f,adj,mask

class MolDataset(Dataset):
    def __init__(self,data):
        self.data=data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
def collate_fn(batch):

    node_f_list, edge_idx_list, edge_f_list, adj_list, mask_list, global_f_list = zip(*batch)

    node_f = torch.tensor(np.stack(node_f_list), dtype=torch.float32)
    edge_f = torch.tensor(np.stack(edge_f_list), dtype=torch.float32)
    adj = torch.tensor(np.stack(adj_list), dtype=torch.float32)
    mask = torch.tensor(np.stack(mask_list), dtype=torch.float32)
    global_f = torch.tensor(np.stack(global_f_list), dtype=torch.float32)

    padded_edge_idx = []

    for e in edge_idx_list:
        e = torch.tensor(e, dtype=torch.long)

        num_e = e.shape[0]

        if num_e >= max_edges:
            e_pad = e[:max_edges]
        else:
            pad = torch.full((max_edges - num_e, 2), -1, dtype=torch.long)
            e_pad = torch.cat([e, pad], dim=0)

        padded_edge_idx.append(e_pad)

    edge_idx = torch.stack(padded_edge_idx)

    return node_f, edge_idx, edge_f, adj, mask, global_f
class MolDataModule:

    def __init__(self, csv_path, batch_size=32, test_size=0.2):

        self.df = pd.read_csv(csv_path)
        self.batch_size = batch_size
        self.test_size = test_size
        self.prep = MolGraphPrep()
        self.scaler = StandardScaler()

    def setup(self):

        data = []

        for row in self.df.itertuples(index=False):

            mol = self.prep.smilesTomol(row.smiles)
            if mol is None:
                continue

            global_f = np.array([row.logP, row.qed, row.SAS], np.float32)
            node_f, edge_idx, edge_f, adj, mask = self.prep.molTograph(mol)

            data.append((node_f, edge_idx, edge_f, adj, mask, global_f))

        train, test = train_test_split(data, test_size=self.test_size)

        g_train = np.array([d[-1] for d in train])
        g_test = np.array([d[-1] for d in test])

        g_train = self.scaler.fit_transform(g_train)
        g_test = self.scaler.transform(g_test)

        train = [(*d[:-1], g_train[i]) for i, d in enumerate(train)]
        test = [(*d[:-1], g_test[i]) for i, d in enumerate(test)]

        self.train_dataset = MolDataset(train)
        self.test_dataset = MolDataset(test)
        train_dataloader= DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn)

        test_dataloader=DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn)
        return train_dataloader,test_dataloader
