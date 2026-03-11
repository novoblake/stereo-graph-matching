import math, cv2, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import from_networkx
from skimage.segmentation import slic
from skimage.color import rgb2gray
from skimage.measure import regionprops
import networkx as nx
import numpy as np
from torch.utils.data import Dataset
import os 
import glob 
from torch_geometric.loader import DataLoader 



class StereoGraphDS(Dataset):
    """
    Folder structure:
    root/train/scene_name/Left/*.png
                 └── Right/*.png
                 └── Disparity/*.png   (ground-truth disparity)
    """
    def __init__(self, root, dmax=64, split='train', transform=None):
        self.root   = root
        self.dmax   = dmax
        self.transform = transform

        # 1. discover all scenes
        self.scenes = sorted([d for d in os.listdir(self.root)
                              if os.path.isdir(os.path.join(self.root, d))])

        # 2. build file lists  (Left names drive the pairing)
        self.samples = []
        for scene in self.scenes:
            left_dir  = os.path.join(self.root, scene, 'Left')
            right_dir = os.path.join(self.root, scene, 'Right')
            disp_dir  = os.path.join(self.root, scene, 'Disparity')

            left_files = sorted(glob.glob(os.path.join(left_dir, '*')))
            for lfile in left_files:
                fname = os.path.basename(lfile)
                rfile = os.path.join(right_dir, fname)
                dfile = os.path.join(disp_dir, fname)
                if os.path.isfile(rfile) and os.path.isfile(dfile):
                    self.samples.append((lfile, rfile, dfile))
                else:
                    print(f'[WARN] missing pair for {lfile} -> skipped')
        print(f'Found {len(self.samples)} stereo pairs in {split} set')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        lpath, rpath, dpath = self.samples[idx]

        # ---- read data ----
        left  = cv2.imread(lpath)[:,:,::-1]          # RGB
        right = cv2.imread(rpath)[:,:,::-1]
        disp  = cv2.imread(dpath, cv2.IMREAD_ANYDEPTH)
        if disp is None:                             # fallback for 8-bit png
            disp = cv2.imread(dpath, 0)
        disp  = disp.astype(np.float32)

        # ---- optional scale/normalise disparity ----
        # (adapt to your GT encoding: here we assume 16-bit px-shift)
        disp /= 256.0   # 64 disparities → 0-255 value range

        # ---- build graph + node target ----
        rag, x, y = build_graph(left, right, self.dmax)   # existing function
        data = from_networkx(rag)
        data.x = x
        data.y = y                      # node-wise GT disparity
        data.fname = os.path.basename(lpath)

        if self.transform:
            data = self.transform(data)
        return data

# ---------- 1.  Fibonacci sampler ----------
def fibonacci_range(dmax):
    """Return Fibonacci numbers ≤ dmax (1,2,3,5,8,…)"""
    fib = [1, 2]
    while fib[-1] + fib[-2] <= dmax:
        fib.append(fib[-1] + fib[-2])
    return fib

# ---------- 2.  Cost between two patches ----------
def patch_cost(patchL, patchR):
    ad  = np.abs(patchL.astype(np.float32) - patchR.astype(np.float32)).mean()
    xor = cv2.cvtColor(patchL, cv2.COLOR_RGB2GRAY) ^ cv2.cvtColor(patchR, cv2.COLOR_RGB2GRAY)
    census = np.sum(xor) / xor.size
    return 0.5 * ad + 0.5 * census

# ---------- 3.  Graph construction ----------
def build_graph(left, right, dmax=64, n_segments=500):
    h, w = left.shape[:2]
    segments = slic(left, n_segments=n_segments, compactness=15, start_label=0)
    rag = nx.Graph()
    # nodes (regionprops expects labels starting at 1)
    for reg in regionprops(segments + 1):
        y, x = reg.centroid
        rag.add_node(reg.label - 1, y=int(y), x=int(x))

    # edges (4-conn)
    for yy in range(h - 1):
        for xx in range(w - 1):
            u, v = segments[yy, xx], segments[yy, xx + 1]
            if u != v:
                rag.add_edge(u, v)
            u, v = segments[yy, xx], segments[yy + 1, xx]
            if u != v:
                rag.add_edge(u, v)

    # ---------- Fibonacci cost ----------
    fib = fibonacci_range(dmax)
    node_count = rag.number_of_nodes()
    node_cost = np.full((node_count, len(fib)), np.inf, dtype=np.float32)

    for i, d in enumerate(fib):
        # skip shift greater than width
        if d >= w:
            continue
        # shift right image leftwards by d (corresponds to disparity d)
        right_shift = np.roll(right, d, axis=1)
        for node, data in rag.nodes(data=True):
            y, x = data['y'], data['x']
            # ensure we have valid coordinates and shifted position
            if x - d < 0 or y - 2 < 0 or y + 3 > h or x - 2 < 0 or x + 3 > w:
                continue
            patchL = left[y - 2:y + 3, x - 2:x + 3]
            patchR = right_shift[y - 2:y + 3, x - 2:x + 3]
            if patchL.shape == patchR.shape == (5, 5, 3):
                node_cost[node, i] = patch_cost(patchL, patchR)

    # Replace +inf with a very large finite value so min/argmin produce finite numbers
    node_cost = np.nan_to_num(node_cost, posinf=1e6, neginf=1e6)

    # best cost and best disparity per node
    best_cost = node_cost.min(axis=1)                     # shape (N,)
    best_idx  = node_cost.argmin(axis=1)                  # index into fib
    best_disp = np.array([fib[i] for i in best_idx], dtype=np.float32)

    # features: [best_cost, best_disp, y, x, area, mean_R, mean_G, mean_B]
    feats = np.zeros((node_count, 8), dtype=np.float32)

    for node, data in rag.nodes(data=True):
        y, x = data['y'], data['x']
        mask = (segments == node)
        area = mask.sum()
        feats[node, 0] = best_cost[node] if np.isfinite(best_cost[node]) else 1e6
        feats[node, 1] = (best_disp[node] / dmax) if np.isfinite(best_disp[node]) else 0.0
        feats[node, 2] = y / h
        feats[node, 3] = x / w
        feats[node, 4] = area / (h * w)
        if area > 0:
            # left[mask] has shape (area, 3)
            feats[node, 5:8] = left[mask].mean(axis=0)
        else:
            # fallback defaults (zero colour)
            feats[node, 5:8] = 0.0

# ---- SAFE NORMALIZATION ----
    mu = np.nanmean(feats, axis=0)
    sigma = np.nanstd(feats, axis=0)

# Replace NaN or zero std with 1
    sigma = np.where((sigma < 1e-6) | np.isnan(sigma), 1.0, sigma)
    mu    = np.where(np.isnan(mu), 0.0, mu)

    feats = (feats - mu) / sigma
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


    # final: convert to torch tensors
    return rag, torch.tensor(feats, dtype=torch.float32), torch.tensor(best_disp, dtype=torch.float32)


# ---------- 4.  Binary Graph-SAGE ----------
class BinarySign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return x.sign()
    @staticmethod
    def backward(ctx, g): return g

class BinSAGE(nn.Module):
    def __init__(self, in_dim, hid, out_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hid)
        self.conv2 = SAGEConv(hid, out_dim)

    def forward(self, x, edge_index):
        # perform forward with binarized weights, without overwriting actual parameters
        # get original weight tensors
        w1_l = self.conv1.lin_l.weight
        w1_r = self.conv1.lin_r.weight
        w2_l = self.conv2.lin_l.weight
        w2_r = self.conv2.lin_r.weight

        # binarize (straight-through) for computation only
        bw1_l = BinarySign.apply(w1_l)
        bw1_r = BinarySign.apply(w1_r)
        bw2_l = BinarySign.apply(w2_l)
        bw2_r = BinarySign.apply(w2_r)

        # temporarily replace via functional linear if needed, or monkey patch .weight.data inside no_grad
        with torch.no_grad():
            orig_w1_l = self.conv1.lin_l.weight.data.clone()
            orig_w1_r = self.conv1.lin_r.weight.data.clone()
            orig_w2_l = self.conv2.lin_l.weight.data.clone()
            orig_w2_r = self.conv2.lin_r.weight.data.clone()

            self.conv1.lin_l.weight.data.copy_(bw1_l)
            self.conv1.lin_r.weight.data.copy_(bw1_r)
            self.conv2.lin_l.weight.data.copy_(bw2_l)
            self.conv2.lin_r.weight.data.copy_(bw2_r)

        # forward
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        # restore weights
        with torch.no_grad():
            self.conv1.lin_l.weight.data.copy_(orig_w1_l)
            self.conv1.lin_r.weight.data.copy_(orig_w1_r)
            self.conv2.lin_l.weight.data.copy_(orig_w2_l)
            self.conv2.lin_r.weight.data.copy_(orig_w2_r)

        return x.squeeze(-1)


# ---------- 6.  Training ----------
def train(model, loader, opt, epochs=30):
    model.train()
    for epoch in range(epochs):
        tot = 0.0
        valid_batches = 0
        for data in loader:
            data = data.to(device)
            opt.zero_grad()
            out = model(data.x, data.edge_index)

            # ---- SAFE LOSS ----
            valid_mask = torch.isfinite(out) & torch.isfinite(data.y)
            if valid_mask.sum() == 0:
                continue  # skip if all invalid

            loss = F.l1_loss(out[valid_mask], data.y[valid_mask])
            if torch.isnan(loss) or torch.isinf(loss):
                print("⚠️ Skipping NaN loss batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # prevent gradient blow-up
            opt.step()
            tot += loss.item()
            valid_batches += 1

        avg_loss = tot / max(valid_batches, 1)
        print(f"epoch {epoch+1:02d}  L1={avg_loss:.4f}")


# ---------- 7.  Inference ----------
def infer(model, left, right, dmax=64):
    model.eval()
    with torch.no_grad():
        rag, x, _ = build_graph(left, right, dmax)
        data = from_networkx(rag)
        data.x = x.to(device)
        disp_nodes = model(data.x, data.edge_index).cpu().numpy()
        # map back to image
        h, w = left.shape[:2]
        segments = slic(left, n_segments=500, compactness=15, start_label=0)
        out = np.zeros_like(segments, dtype=np.float32)
        for node, d in enumerate(disp_nodes):
            out[segments == node] = d * dmax   # de-normalise
    return out

# ------------------ demo ------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds  = StereoGraphDS(root= r'D:\WHU_stereo_dataset\train', split='train', dmax=64)
    val_ds    = StereoGraphDS(root= r'D:\WHU_stereo_dataset\train', split='val',   dmax=64)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False)

    model = BinSAGE(8, 128, 1).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-4)
    torch.autograd.set_detect_anomaly(True)                 # helps pinpoint NaN/inf in backward
    train(model, train_loader, opt, epochs=30)
    # quick test
    l, r = cv2.imread("data/left/000001.png")[:,:,::-1], cv2.imread("data/right/000001.png")[:,:,::-1]
    disp = infer(model, l, r)
    cv2.imwrite("disp_fib.png", (disp/disp.max()*255).astype(np.uint8))