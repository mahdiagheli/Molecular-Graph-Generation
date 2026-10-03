import torch
from rdkit import Chem
from rdkit.Chem import Draw
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from preprocessingTorch import MolDataModule
from GNN_VAETorch import GraphVAE
from pathlib import Path 
import datetime
import matplotlib.pyplot as plt

class MoleculeSamplerRDKit:
    def __init__(self, model, latent_dim, max_atom=50, sample_every=5, device="cuda"):
        self.model = model
        self.latent_dim = latent_dim
        self.sample_every = sample_every
        self.max_atom = max_atom
        self.device = device

    @torch.no_grad()
    def sample(self, epoch):
        if (epoch + 1) % self.sample_every != 0:
            return

        print(f"\n--- Sampling molecule at epoch {epoch+1} ---")

        self.model.eval()

        z = torch.randn(1, self.latent_dim).to(self.device)
        node_f, adj_prob, edge_f, global_f = self.model.decoder(z)

        node_f = node_f[0].cpu().numpy()
        adj_prob = adj_prob[0].cpu().numpy()

        mol = Chem.RWMol()
        atom_map = {}

        for i in range(self.max_atom):
            if node_f[i].sum() > 0.1:
                atomic_num = int(round(node_f[i][0]))
                if 1 <= atomic_num <= 100:
                    idx = mol.AddAtom(Chem.Atom(atomic_num))
                    atom_map[i] = idx

        for i in range(self.max_atom):
            for j in range(i + 1, self.max_atom):
                if i in atom_map and j in atom_map:
                    if adj_prob[i, j] > 0.5:
                        mol.AddBond(atom_map[i], atom_map[j], Chem.BondType.SINGLE)

        mol = mol.GetMol()
        Chem.SanitizeMol(mol)

        img = Draw.MolToImage(mol, size=(300,300))
        img.save(f"sample_epoch_{epoch+1}.png")

        print("Molecule saved.")


def train_model(model, train_loader, test_loader, epochs, lr):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    log_dir = "logs/vae/" + datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=log_dir)

    sampler = MoleculeSamplerRDKit(model, latent_dim=128, device=device)
    metric_keys = ["loss", "node_loss", "adj_loss", "edge_loss", "global_loss", "kl_loss"]

    best_loss = float("inf")
    history = {
        "train": {k: [] for k in metric_keys},
        "test": {k: [] for k in metric_keys},
    }

    for epoch in range(epochs):

        model.train()
        train_sums = {k: 0.0 for k in metric_keys}

        loop = tqdm(train_loader)

        for batch in loop:
            batch = tuple(x.to(device) for x in batch)
            metrics = model.train_step(batch, optimizer)
            for k in metric_keys:
                train_sums[k] += metrics[k].item()
            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(
                loss=f"{metrics['loss'].item():.4f}",
                node=f"{metrics['node_loss'].item():.4f}",
                adj=f"{metrics['adj_loss'].item():.4f}",
                edge=f"{metrics['edge_loss'].item():.4f}",
                glob=f"{metrics['global_loss'].item():.4f}",
                kl=f"{metrics['kl_loss'].item():.4f}",
            )

        avg_train = {k: train_sums[k] / len(train_loader) for k in metric_keys}
        avg_train_loss = avg_train["loss"]
        for k in metric_keys:
            history["train"][k].append(avg_train[k])

        for k in metric_keys:
            writer.add_scalar(f"Train/{k}", avg_train[k], epoch)

        model.eval()
        test_sums = {k: 0.0 for k in metric_keys}

        with torch.no_grad():
            for batch in test_loader:
                batch = tuple(x.to(device) for x in batch)
                metrics = model.test_step(batch)
                for k in metric_keys:
                    test_sums[k] += metrics[k].item()


        avg_test = {k: test_sums[k] / len(test_loader) for k in metric_keys}
        avg_test_loss = avg_test["loss"]
        for k in metric_keys:
            history["test"][k].append(avg_test[k])
        for k in metric_keys:
            writer.add_scalar(f"Test/{k}", avg_test[k], epoch)

        print(
            f"\nEpoch {epoch+1}\n"
            f"Train | loss={avg_train['loss']:.4f} node={avg_train['node_loss']:.4f} adj={avg_train['adj_loss']:.4f} "
            f"edge={avg_train['edge_loss']:.4f} global={avg_train['global_loss']:.4f} kl={avg_train['kl_loss']:.4f}\n"
            f"Test  | loss={avg_test['loss']:.4f} node={avg_test['node_loss']:.4f} adj={avg_test['adj_loss']:.4f} "
            f"edge={avg_test['edge_loss']:.4f} global={avg_test['global_loss']:.4f} kl={avg_test['kl_loss']:.4f}"
        )

        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            torch.save(model.state_dict(), "vae_model.pt")
            print("Model saved.")


        sampler.sample(epoch)

    writer.close()
    return history


def plot_all_losses(history):
    metric_keys = ["loss", "node_loss", "adj_loss", "edge_loss", "global_loss", "kl_loss"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for i, key in enumerate(metric_keys):
        ax = axes[i]
        ax.plot(history["train"][key], label=f"train_{key}")
        ax.plot(history["test"][key], label=f"test_{key}")
        ax.set_title(key)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.show()

base = Path(__file__).parent.parent

df= MolDataModule(
    csv_path=base / "data/zinc.csv",
    batch_size=64,
    test_size=0.2
)
train_loader,test_loader=df.setup()
model=GraphVAE(latent_dim=128)
history = train_model(
    model,
    train_loader,
    test_loader,
    epochs=30,
    lr=1e-3
)
plot_all_losses(history)
