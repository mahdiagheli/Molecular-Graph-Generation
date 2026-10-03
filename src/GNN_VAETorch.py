import torch
import torch.nn as nn

max_atom=50
node_feature_dim=10
edge_feature_dim=6
global_feature_dim=3

class GNNEncoder(nn.Module):
    def __init__(self,latent_dim,hiddem_dim=256,drop_rate=0.3):
        super().__init__()

        self.node_l1=nn.Linear(node_feature_dim, hiddem_dim)
        self.node_bn1=nn.BatchNorm1d(hiddem_dim)
        self.node_ac1=nn.LeakyReLU(0.1)
        self.node_do1=nn.Dropout(drop_rate)

        self.node_l2=nn.Linear(hiddem_dim, hiddem_dim)
        self.node_bn2=nn.BatchNorm1d(hiddem_dim)
        self.node_ac2=nn.LeakyReLU(0.1)
        self.node_do2=nn.Dropout(drop_rate)

        self.edg_l1=nn.Linear(edge_feature_dim, hiddem_dim)
        self.edg_bn1=nn.BatchNorm1d(hiddem_dim)
        self.edg_ac1=nn.LeakyReLU(0.1)
        self.edg_do1=nn.Dropout(drop_rate)

        self.edg_l2=nn.Linear(hiddem_dim, hiddem_dim)
        self.edg_bn2=nn.BatchNorm1d(hiddem_dim)
        self.edg_ac2=nn.LeakyReLU(0.1)
        self.edg_do2=nn.Dropout(drop_rate)

        self.msg_l=nn.Linear(hiddem_dim * 2, hiddem_dim)
        self.msg_ac=nn.LeakyReLU(0.1)

        self.graph_l1=nn.Linear(hiddem_dim + global_feature_dim, hiddem_dim)
        self.graph_bn1=nn.LayerNorm(hiddem_dim)
        self.graph_ac1=nn.LeakyReLU(0.1)
        self.graph_do1=nn.Dropout(drop_rate)

        self.graph_l2=nn.Linear(hiddem_dim, hiddem_dim)
        self.graph_bn2=nn.LayerNorm(hiddem_dim)
        self.graph_ac2=nn.LeakyReLU(0.1)
        self.graph_do2=nn.Dropout(drop_rate)

        self.z_mean=nn.Linear(hiddem_dim, latent_dim)
        self.z_logvar=nn.Linear(hiddem_dim, latent_dim)

        self._init_he(self.node_l1)
        self._init_he(self.edg_l1)
        self._init_he(self.msg_l)
        self._init_he(self.graph_l1)

    @staticmethod
    def _init_he(layer):
        nn.init.kaiming_uniform_(layer.weight, a=0.1, nonlinearity="leaky_relu")
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    @staticmethod
    def _apply_bn_3d(x, bn_layer):
        b, n, c = x.shape
        x = x.reshape(b * n, c)
        x = bn_layer(x)
        return x.reshape(b, n, c)

    def forward(self,node_f,edg_idx,edge_f,adj,mask,global_f):
        x=self.node_l1(node_f)
        x=self._apply_bn_3d(x, self.node_bn1)
        x=self.node_ac1(x)
        x=self.node_do1(x)

        x=self.node_l2(x)
        x=self._apply_bn_3d(x, self.node_bn2)
        x=self.node_ac2(x)
        x=self.node_do2(x)

        e=self.edg_l1(edge_f)
        e=self._apply_bn_3d(e.reshape(e.shape[0], -1, e.shape[-1]), self.edg_bn1).reshape_as(e)
        e=self.edg_ac1(e)
        e=self.edg_do1(e)

        e=self.edg_l2(e)
        e=self._apply_bn_3d(e.reshape(e.shape[0], -1, e.shape[-1]), self.edg_bn2).reshape_as(e)
        e=self.edg_ac2(e)
        e=self.edg_do2(e)

        for _ in range(2):
            valid_mask = torch.ne(edg_idx[:, :, 0], -1)
            valid_edges = valid_mask.nonzero(as_tuple=False).to(dtype=torch.long)
            src = edg_idx[:, :, 0][valid_mask]
            dst = edg_idx[:, :, 1][valid_mask]
            batch = valid_edges[:, 0]


            src_feat = x[batch, src]
            dst_feat = x[batch, dst]
            edge_feat = e[batch, src, dst]

            msg_input = torch.cat([src_feat, edge_feat], dim=-1)
            msg = self.msg_l(msg_input)
            msg = self.msg_ac(msg)


            agg = torch.zeros_like(x)
            agg.index_put_((batch, dst), msg, accumulate=True)
            x = x + agg



        x = x * mask.unsqueeze(-1)
        x = torch.sum(x, dim=1)


        x = torch.cat([x, global_f], dim=-1)

        x=self.graph_l1(x)
        x=self.graph_bn1(x)
        x=self.graph_ac1(x)
        x=self.graph_do1(x)

        x=self.graph_l2(x)
        x=self.graph_bn2(x)
        x=self.graph_ac2(x)
        x=self.graph_do2(x)

        z_mean,z_logvar=self.z_mean(x),self.z_logvar(x)
        return z_mean,z_logvar

class sampling(nn.Module):
    def forward(self,inputs):
        z_mean,z_logvar=inputs
        eps=torch.randn_like(z_mean)
        return z_mean+torch.exp(0.5*z_logvar)*eps

class GNNDecoder(nn.Module):
    def __init__(self,drop_rate=0.3):
        super().__init__()
        self.l1=nn.LazyLinear(256)
        self.bn1=nn.BatchNorm1d(256)
        self.ac1=nn.LeakyReLU(0.1)
        self.do1=nn.Dropout(drop_rate)

        self.l2=nn.Linear(256, 128)
        self.bn2=nn.BatchNorm1d(128)
        self.ac2=nn.LeakyReLU(0.1)
        self.do2=nn.Dropout(drop_rate)

        self.l3=nn.Linear(128, 64)
        self.bn3=nn.BatchNorm1d(64)
        self.ac3=nn.LeakyReLU(0.1)
        self.do3=nn.Dropout(drop_rate)

        self.node_out = nn.Linear(64, max_atom * node_feature_dim)
        self.adj_out = nn.Linear(64, max_atom * max_atom)
        self.edge_out = nn.Linear(64, max_atom * max_atom * edge_feature_dim)
        self.global_out = nn.Linear(64, global_feature_dim)

    def forward(self,z):
        x=self.l1(z)
        x=self.bn1(x)
        x=self.ac1(x)
        x=self.do1(x)

        x=self.l2(x)
        x=self.bn2(x)
        x=self.ac2(x)
        x=self.do2(x)

        x=self.l3(x)
        x=self.bn3(x)
        x=self.ac3(x)
        x=self.do3(x)

        node_f=self.node_out(x)
        adj_logits=self.adj_out(x)
        edge_f=self.edge_out(x)
        global_f=self.global_out(x)
        node_f = node_f.reshape(-1, max_atom, node_feature_dim)
        adj_prob = adj_logits.reshape(-1, max_atom, max_atom)
        adj_prob = torch.sigmoid(adj_prob)
        adj_prob = 0.5 * (adj_prob + adj_prob.transpose(1, 2))
        adj_prob *= (1 - torch.eye(max_atom, device=adj_prob.device, dtype=adj_prob.dtype))

        edge_f = edge_f.reshape(-1, max_atom, max_atom, edge_feature_dim)
        edge_f = edge_f * adj_prob.unsqueeze(-1)
        return node_f,adj_prob,edge_f,global_f


class GraphVAE(nn.Module):
    def __init__(self,latent_dim,beta=0.1):
        super().__init__()
        self.encoder=GNNEncoder(latent_dim)
        self.sampling=sampling()
        self.decoder=GNNDecoder()
        self.beta=beta

    def forward(self,inputs):
        node_f,edge_idx,edge_f,adj,mask,global_f=inputs
        z_mean,z_logvar=self.encoder(node_f,edge_idx,edge_f,adj,mask,global_f)
        z=self.sampling((z_mean,z_logvar))
        node_hat,adj_hat,edg_hat,global_hat=self.decoder(z)
        return node_hat,adj_hat,edg_hat,global_hat,z_mean,z_logvar
    
    def compute_loss(self, inputs):
        node_f, edge_idx, edge_f, adj, mask, global_f = inputs
        node_hat, adj_hat, edg_hat, global_hat, z_mean, z_logvar = self.forward(inputs)
        
   
        node_mask = mask.unsqueeze(-1) 
        node_loss = torch.mean(
            torch.sum(torch.square(node_f - node_hat) * node_mask, dim=(1,2))
        )

        mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)
        mask_edge = mask_2d.unsqueeze(-1)
        if edg_hat.dim() == 3: 
            edg_hat = edg_hat.unsqueeze(-1)
        edge_loss = torch.mean(
            torch.sum(torch.square(edge_f - edg_hat) * mask_edge, dim=(1,2,3)))
        eps = 1e-7
        adj_loss = (- adj * torch.log(adj_hat + eps)- (1.0 - adj) * torch.log(1.0 - adj_hat + eps))

        if adj_loss.dim() == 4:
            adj_loss = adj_loss.squeeze(-1)
        adj_loss = torch.sum(adj_loss * mask_2d, dim=(1,2))
        adj_loss = torch.mean(adj_loss)
        global_loss = torch.mean(torch.sum(torch.square(global_f - global_hat), dim=1))
        node_loss=1/191*node_loss
        edge_loss=1/72.6*edge_loss
        adj_loss=1/75*adj_loss
        global_loss=1/2.32*global_loss
        recon_loss = node_loss + edge_loss + adj_loss + global_loss
        kl = -0.5 * torch.mean(
            torch.sum(1 + z_logvar - torch.square(z_mean) - torch.exp(z_logvar), dim=1)
        )
        loss = recon_loss + self.beta * kl

        return loss, node_loss, adj_loss, edge_loss, global_loss, kl

    def train_step(self, data, optimizer):
        inputs = data
        self.train()
        optimizer.zero_grad()
        loss, n_loss, a_loss, e_loss, g_loss, kl = self.compute_loss(inputs)
        loss.backward()
        optimizer.step()
        return {"loss": loss, "node_loss": n_loss, "adj_loss": a_loss, "edge_loss": e_loss, "global_loss": g_loss, "kl_loss": kl}

    def test_step(self, data):
        inputs = data
        self.eval()
        with torch.no_grad():
            loss, n_loss, a_loss, e_loss, g_loss, kl = self.compute_loss(inputs)
        return {"loss": loss, "node_loss": n_loss, "adj_loss": a_loss, "edge_loss": e_loss, "global_loss": g_loss, "kl_loss": kl}





