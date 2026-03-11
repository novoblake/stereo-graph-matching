"""
Enhanced stereo superpixel-graph model for aerial stereo.
Features included:
- Atrous/ASPP backbone feature extractor (ResNet-based)
- Superpixel RAG construction (slic + regionprops)
- Node aggregation of multi-scale features
- Candidate disparities = Fibonacci U local dense band
- Cosine similarity + groupwise correlation + patch-cost fusion
- Differentiable soft-argmin for node disparity
- Graph attention refinement (GAT)
- Pixel-level refinement U-Net (optional, lightweight)
- Training loop with supervised L1 + distribution loss

Notes:
- This file is a runnable sketch that focuses on architecture and integration.
- For heavy training, adapt dataloader batching, mixed precision, and distributed training.
- Dependencies: torch, torchvision, torch_geometric, scikit-image, opencv-python, networkx, numpy

"""

import os
import glob
import math
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50
from skimage.segmentation import slic
from skimage.measure import regionprops
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GATConv
from torch.utils.data import Dataset, DataLoader

# -------------------- Utility functions --------------------

def fibonacci_range(dmax: int) -> List[int]:
    fib = [1, 2]
    while fib[-1] + fib[-2] <= dmax:
        fib.append(fib[-1] + fib[-2])
    return fib


def compute_patch_cost(patchL: np.ndarray, patchR: np.ndarray) -> float:
    # keep original ad + census inspired cost but scaled
    ad = np.abs(patchL.astype(np.float32) - patchR.astype(np.float32)).mean()
    grayL = cv2.cvtColor(patchL, cv2.COLOR_RGB2GRAY)
    grayR = cv2.cvtColor(patchR, cv2.COLOR_RGB2GRAY)
    xor = cv2.bitwise_xor(grayL, grayR)
    census = xor.astype(np.float32).sum() / xor.size
    return 0.5 * ad + 0.5 * census


# -------------------- Dataset (returns filepaths) --------------------
class StereoPathDataset(Dataset):
    def __init__(self, root: str, split: str = 'train'):
        self.root = root
        left_dir  = os.path.join(self.root, 'Left')
        right_dir = os.path.join(self.root, 'Right')
        disp_dir  = os.path.join(self.root, 'Disparity')

        self.samples = []
        left_files = sorted(glob.glob(os.path.join(left_dir, '*')))
        for lfile in left_files:
            fname = os.path.basename(lfile)
            rfile = os.path.join(right_dir, fname)
            dfile = os.path.join(disp_dir, fname)
            if os.path.isfile(rfile) and os.path.isfile(dfile):
                self.samples.append((lfile, rfile, dfile))
            else:
                print(f'[WARN] missing pair for {lfile} -> skipped')

        print(f'✅ Found {len(self.samples)} stereo pairs in {split} set')


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# -------------------- ASPP block (Atrous Spatial Pyramid) --------------------
class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rates=(1, 2, 4)):
        super().__init__()
        self.convs = nn.ModuleList()
        for r in rates:
            padding = r
            self.convs.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=padding, dilation=r))
        self.project = nn.Conv2d(len(rates) * out_ch, out_ch, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        outs = [F.relu(conv(x)) for conv in self.convs]
        out = torch.cat(outs, dim=1)
        out = self.project(out)
        out = self.bn(out)
        return F.relu(out)


# -------------------- Feature backbone --------------------
class FeatureBackbone(nn.Module):
    def __init__(self, pretrained=True, out_dim=128):
        super().__init__()
        # use ResNet50 trunk and take features from layer3
        r50 = resnet50(pretrained=pretrained)
        # remove avgpool and fc
        self.layer0 = nn.Sequential(r50.conv1, r50.bn1, r50.relu, r50.maxpool)
        self.layer1 = r50.layer1
        self.layer2 = r50.layer2
        self.layer3 = r50.layer3
        # ASPP on top of layer3
        self.aspp = ASPP(in_ch=1024, out_ch=out_dim, rates=(1, 2, 4))

    def forward(self, x):
        # x: [B,3,H,W]  (we expect B=1 in current pipeline, but keep B)
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        features = self.aspp(x3)  # [B, out_dim, H', W']
        return features


# -------------------- Simple pixel refinement U-Net --------------------
class PixelRefineUNet(nn.Module):
    def __init__(self, in_ch=4, base=32):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base, 3, padding=1), nn.ReLU(), nn.Conv2d(base, base, 3, padding=1), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(base, base * 2, 3, padding=1), nn.ReLU(), nn.Conv2d(base * 2, base * 2, 3, padding=1), nn.ReLU())
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = nn.Sequential(nn.Conv2d(base * 3, base, 3, padding=1), nn.ReLU(), nn.Conv2d(base, base, 3, padding=1), nn.ReLU())
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, img, disp, conf):
        # img: [B,3,H,W], disp: [B,1,H,W], conf: [B,1,H,W]
        x = torch.cat([img, disp, conf], dim=1)
        e1 = self.enc1(x)
        p = self.pool(e1)
        e2 = self.enc2(p)
        u = self.up(e2)
        # concat with e1
        cat = torch.cat([u, e1], dim=1)
        d = self.dec1(cat)
        out = self.out(d)
        return out


# -------------------- Graph Stereo Model --------------------
class StereoGraphNet(nn.Module):
    def __init__(self, dmax=128, node_feat_dim=128, candidate_local_radius=5, device='cpu'):
        super().__init__()
        self.device = device
        self.dmax = dmax
        self.backbone = FeatureBackbone(pretrained=True, out_dim=node_feat_dim)
        # compress node feature after aggregation
        self.node_mlp = nn.Sequential(nn.Linear(node_feat_dim + 3, 256), nn.ReLU(), nn.Linear(256, 128))
        # cost mlp
        self.cost_mlp = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 1))
        # GAT refinement
        self.gat1 = GATConv(128 + 1 + 1, 128, heads=4)  # node_feat + disp + conf
        self.gat2 = GATConv(128 * 4, 64, heads=4)
        self.reg_head = nn.Sequential(nn.Linear(64 * 4, 64), nn.ReLU(), nn.Linear(64, 2))  # delta_disp, log_conf
        # pixel refinement
        self.pixel_refine = PixelRefineUNet(in_ch=5, base=32)

    # ---------- image reading and preproc ----------
    @staticmethod
    def read_image(path: str) -> np.ndarray:
        im = cv2.imread(path)[:, :, ::-1]  # BGR->RGB
        im = im.astype(np.float32) / 255.0
        return im

    @staticmethod
    def read_disp(path: str) -> np.ndarray:
        disp = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        if disp is None:
            disp = cv2.imread(path, 0)
        disp = disp.astype(np.float32) / 256.0
        return disp

    # ---------- build superpixel RAG and node metadata ----------
    def build_rag(self, left_img: np.ndarray, n_segments=400, compactness=15):
        h, w = left_img.shape[:2]
        segments = slic((left_img * 255).astype(np.uint8), n_segments=n_segments, compactness=compactness, start_label=0)
        rag = nx.Graph()
        masks = []
        for reg in regionprops(segments + 1):
            y, x = reg.centroid
            node_id = reg.label - 1
            rag.add_node(node_id, y=int(y), x=int(x))
            masks.append((segments == node_id))
        # adjacency via 4-neighbor scanning (fast approach)
        for yy in range(h - 1):
            for xx in range(w - 1):
                u, v = segments[yy, xx], segments[yy, xx + 1]
                if u != v:
                    rag.add_edge(u, v)
                u, v = segments[yy, xx], segments[yy + 1, xx]
                if u != v:
                    rag.add_edge(u, v)
        return rag, segments, masks

    # ---------- aggregate pixel features to nodes ----------
    def aggregate_node_features(self, feat_map: torch.Tensor, segments: np.ndarray, masks: List[np.ndarray]) -> torch.Tensor:
        # feat_map: [1, C, Hf, Wf] ; segments: HxW in original image coords
        # We'll map segments to feature resolution by simple downscale
        _, C, Hf, Wf = feat_map.shape
        H, W = segments.shape
        # resize segments to Hf x Wf via nearest
        seg_small = cv2.resize(segments.astype(np.int32), (Wf, Hf), interpolation=cv2.INTER_NEAREST)
        # move features to cpu numpy for aggregation simplicity
        B, C, Hf, Wf = feat_map.shape
        fmap = feat_map[0]  # keep as torch.Tensor
        node_feats = []
        seg_small = torch.from_numpy(cv2.resize(segments.astype(np.int32), (Wf, Hf), interpolation=cv2.INTER_NEAREST)).to(self.device)

        for node_id in range(len(masks)):
            mask_small = (seg_small == node_id)
            if mask_small.sum() == 0:
                mean_feat = torch.zeros(C, device=self.device)
            else:
                vals = fmap[:, mask_small]
                mean_feat = vals.mean(dim=1)
            node_feats.append(mean_feat)
        node_feats = torch.stack(node_feats, dim=0)  # shape [N,C]

        return node_feats

    # ---------- compute candidate right features for each disparity candidate ----------
    def gather_right_features_for_candidates(self, right_feat_map: torch.Tensor, segments: np.ndarray, masks: List[np.ndarray], candidates: List[int]) -> torch.Tensor:
        # returns tensor shaped (N, C, K)
        # We'll sample right features using centroid positions projected by disparity shift (approx)
        _, C, Hf, Wf = right_feat_map.shape
        # compute centroid map in feature coords
        N = len(masks)
        node_centroids = []
        H, W = segments.shape
        # compute segment centroids in original coords then to feature coords
        for node_id, mask in enumerate(masks):
            ys, xs = np.nonzero(mask)
            if len(ys) == 0:
                node_centroids.append((0, 0))
            else:
                cy = int(np.round(ys.mean()))
                cx = int(np.round(xs.mean()))
                # map to feature coords
                fy = int(np.round(cy * (Hf / H)))
                fx = int(np.round(cx * (Wf / W)))
                fy = max(0, min(Hf - 1, fy))
                fx = max(0, min(Wf - 1, fx))
                node_centroids.append((fy, fx))
        # get feature map as numpy
        fmap = right_feat_map[0].detach().cpu().numpy()  # C,Hf,Wf
        out = np.zeros((N, C, len(candidates)), dtype=np.float32)
        for i, (fy, fx) in enumerate(node_centroids):
            for k, d in enumerate(candidates):
                # approximate epipolar shift in feature coords: assume disparity in pixels -> shift in image coords -> scaled to feature coords
                # since right image shifted left by d px corresponds to feature shift ~ d * (Wf / W)
                # we approximate using fx - shift
                # this is a heuristic; for more accuracy use intrinsics/rectification mapping
                shift = int(round(d * (Wf / segments.shape[1])))
                rx = fx - shift
                if rx < 0 or rx >= Wf:
                    # out-of-bounds -> zero vector
                    out[i, :, k] = 0.0
                else:
                    out[i, :, k] = fmap[:, fy, rx]
        return torch.tensor(out, dtype=torch.float32, device=self.device)  # N,C,K

    # ---------- cost computation (cosine + correlation + patch) ----------
    def compute_cost_matrix(self, node_feats_left: torch.Tensor, right_feats_candidates: torch.Tensor, left_img: np.ndarray, segments: np.ndarray, masks: List[np.ndarray], candidates: List[int]) -> torch.Tensor:
        # node_feats_left: N,C ; right_feats_candidates: N,C,K
        N, C = node_feats_left.shape
        K = len(candidates)
        # cosine similarity
        left_exp = node_feats_left.unsqueeze(-1).expand(-1, -1, K)  # N,C,K
        cosine = F.cosine_similarity(left_exp, right_feats_candidates, dim=1)  # N,K
        # correlation (dot)
        corr = (left_exp * right_feats_candidates).sum(dim=1)  # N,K
        # patch cost (computed on CPU using small patches) - optional and slower
        patch_costs = np.zeros((N, K), dtype=np.float32)
        h, w = left_img.shape[:2]
        for i, mask in enumerate(masks):
            ys, xs = np.nonzero(mask)
            if len(ys) == 0:
                patch_costs[i, :] = 1e6
                continue
            cy = int(round(ys.mean()))
            cx = int(round(xs.mean()))
            for k, d in enumerate(candidates):
                if cx - d < 2 or cx + 3 > w - 1 or cy - 2 < 0 or cy + 3 > h - 1:
                    patch_costs[i, k] = 1e6
                else:
                    patchL = (left_img[cy - 2:cy + 3, cx - 2:cx + 3] * 255).astype(np.uint8)
                    patchR = (np.roll(left_img, d, axis=1)[cy - 2:cy + 3, cx - 2:cx + 3] * 255).astype(np.uint8)
                    if patchL.shape == patchR.shape == (5, 5, 3):
                        patch_costs[i, k] = compute_patch_cost(patchL, patchR)
                    else:
                        patch_costs[i, k] = 1e6
        patch_costs_t = torch.tensor(patch_costs, dtype=torch.float32, device=self.device)
        # combine
        # normalize each channel
        cosine_n = (cosine - cosine.mean(dim=1, keepdim=True)) / (cosine.std(dim=1, keepdim=True) + 1e-6)
        corr_n = (corr - corr.mean(dim=1, keepdim=True)) / (corr.std(dim=1, keepdim=True) + 1e-6)
        patch_n = (patch_costs_t - patch_costs_t.mean(dim=1, keepdim=True)) / (patch_costs_t.std(dim=1, keepdim=True) + 1e-6)
        cost_inputs = torch.stack([cosine_n, corr_n, patch_n], dim=-1)  # N,K,3
        # apply small MLP per candidate
        Np = N * K
        cost_flat = cost_inputs.view(Np, 3)
        logits = self.cost_mlp(cost_flat).view(N, K)
        return logits  # raw cost logits (lower = better)

    # ---------- soft argmin to produce node disparities and distributions ----------
    def soft_argmin(self, cost_logits: torch.Tensor, candidates: List[int], temp: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        # cost_logits: N,K  (lower better). We convert to probs
        probs = F.softmax(-cost_logits / temp, dim=1)  # N,K
        d_candidates = torch.tensor(candidates, dtype=torch.float32, device=self.device).unsqueeze(0)  # 1,K
        d_hat = (probs * d_candidates).sum(dim=1)  # N
        return d_hat, probs

    # ---------- full forward pipeline given image paths ----------
    def forward_from_paths(self, left_path: str, right_path: str, disp_path: str) -> dict:
        left = self.read_image(left_path)
        right = self.read_image(right_path)
        gt_disp = self.read_disp(disp_path)
        h, w = left.shape[:2]
        # build rag
        rag, segments, masks = self.build_rag(left, n_segments=800)
        node_count = rag.number_of_nodes()
        # backbone features
        
            # convert to torch tensor and to device
        left_t = torch.tensor(left.transpose(2, 0, 1)[None], dtype=torch.float32, device=self.device)
        right_t = torch.tensor(right.transpose(2, 0, 1)[None], dtype=torch.float32, device=self.device)
        left_feats = self.backbone(left_t)  # [1,C,Hf,Wf]
        right_feats = self.backbone(right_t)
        # aggregate
        node_feats = self.aggregate_node_features(left_feats, segments, masks)  # N,C
        # append simple geometric features (y/h, x/w, area)
        geom = []
        for node_id, data in rag.nodes(data=True):
            y, x = data['y'], data['x']
            area = masks[node_id].sum()
            geom.append([y / h, x / w, area / (h * w)])
        geom = torch.tensor(np.stack(geom, axis=0), dtype=torch.float32, device=self.device)
        node_concat = torch.cat([node_feats, geom], dim=1)
        node_emb = self.node_mlp(node_concat)  # N,128
        # candidate disparities
        fib = fibonacci_range(self.dmax)
        # we compute a local dense band around each fib value: we'll create combined candidate list
        # For efficiency choose small K: take unique sorted union of fib and local bands around each fib within [0,dmax]
        candidates_set = set()
        local_r = 3  # local radius
        for f in fib:
            candidates_set.add(f)
            for dd in range(max(0, f - local_r), min(self.dmax, f + local_r + 1)):
                candidates_set.add(dd)
        candidates = sorted([int(x) for x in candidates_set if x >= 0 and x < w])
        # get right features for all candidates
        right_cand_feats = self.gather_right_features_for_candidates(right_feats, segments, masks, candidates)  # N,C,K
        # compute cost logits
        cost_logits = self.compute_cost_matrix(node_emb, right_cand_feats, left, segments, masks, candidates)  # N,K
        # soft-argmin
        d_hat, probs = self.soft_argmin(cost_logits, candidates)
        # initial confidence as max prob
        conf = probs.max(dim=1).values
        # GNN refinement
        # create torch_geometric data from rag
        data = from_networkx(rag)
        data.x = node_emb
        data.edge_index = data.edge_index.to(self.device)
        # append disp and conf as node features
        disp_feat = d_hat.unsqueeze(-1) / float(self.dmax)
        conf_feat = conf.unsqueeze(-1)
        x_in = torch.cat([data.x.to(self.device), disp_feat, conf_feat], dim=1)
        x = F.elu(self.gat1(x_in, data.edge_index))
        x = self.gat2(x, data.edge_index)
        # reg head
        node_out = self.reg_head(x)
        delta = node_out[:, 0]
        log_conf = node_out[:, 1]
        d_refined = d_hat + delta
        conf_refined = torch.sigmoid(-log_conf)  # higher -> more confident (learned mapping)

        # project to pixel map
        seg_arr = segments 
        seg_t = torch.from_numpy(segments.astype(np.int64)).to(self.device)
        Hs, Ws = seg_arr.shape
        out_disp_t = torch.zeros((Hs, Ws), dtype=torch.float32, device=self.device)
        for node_id in range(node_count):
            mask = (seg_t == node_id)
            if mask.any():
                out_disp_t[mask] = d_refined[node_id].clamp(min=0.0, max=float(self.dmax))
                


        # pixel refinement (keep tensors on device)
        left_tensor = torch.tensor(left.transpose(2, 0, 1)[None], dtype=torch.float32, device=self.device)
        disp_tensor = out_disp_t.unsqueeze(0).unsqueeze(0) / float(self.dmax)  # [1,1,H,W]
        # conf_refined is tensor on device; expand to pixel map similarly
        conf_map_t = torch.zeros((Hs, Ws), dtype=torch.float32, device=self.device)
        for nid in range(node_count):
            mask = (seg_t == nid)
            if mask.any():
                conf_map_t[mask] = conf_refined[nid]
        conf_tensor = conf_map_t.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

        refined = self.pixel_refine(left_tensor, disp_tensor, conf_tensor)  # [1,1,H,W]
        refined_disp_t = refined[0, 0] * float(self.dmax)  # still a tensor on device

        # ---------- return: TENSORS during training, NUMPY for eval ----------
        if self.training:
            return {
                'node_disp_raw': d_hat,                 # tensor (requires_grad)
                'node_disp_refined': d_refined,        # tensor (requires_grad)
                'pixel_disp_initial': out_disp_t,      # tensor (on device)
                'pixel_disp_refined': refined_disp_t,  # tensor (on device)
                'gt_disp': torch.from_numpy(gt_disp).to(self.device),  # tensor (no grad)
                'segments': segments                   # keep numpy for mask ops/visualization
            }
        else:
            # Detach & move to cpu numpy for inference / visualization
            return {
                'node_disp_raw': d_hat.detach().cpu().numpy(),
                'node_disp_refined': d_refined.detach().cpu().numpy(),
                'pixel_disp_initial': out_disp_t.detach().cpu().numpy(),
                'pixel_disp_refined': refined_disp_t.detach().cpu().numpy(),
                'gt_disp': gt_disp,
                'segments': segments
            }



# -------------------- Training loop --------------------
def train_loop(train_root, val_root, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    ds = StereoPathDataset(train_root, split='train')
    val_ds = StereoPathDataset(val_root, split='val')
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

    model = StereoGraphNet(dmax=128, node_feat_dim=128, device=device).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_loss = float("inf")

    train_logs = []  # store (epoch, loss)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_file = f"training_log_{timestamp}.csv"

    # Write CSV header
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train_L1_Loss"])

    print(f"🧾 Logging training progress to {log_file}")


    for epoch in range(1, 31):  # 30 epochs
        model.train()
        total_loss = 0.0
        count = 0

        for (lpath, rpath, dpath) in loader:
            lpath, rpath, dpath = lpath[0], rpath[0], dpath[0]
            out = model.forward_from_paths(lpath, rpath, dpath)
            
            # compute node-level loss
            segments = out['segments']
            gt = out['gt_disp']
            node_gts = []
            N = np.max(segments) + 1
            for nid in range(N):
                mask = (segments == nid)
                vals = gt[mask]
                #node_gts.append(np.nanmean(vals) if vals.size > 0 else np.nan)
                vals_np = vals.cpu().numpy() if isinstance(vals, torch.Tensor) else vals
                node_gts.append(np.nanmean(vals_np) if vals_np.size > 0 else np.nan)
            node_gts = np.array(node_gts, dtype=np.float32)
            
            pred_nodes = out['node_disp_refined']
            mask_valid = np.isfinite(node_gts)
            if mask_valid.sum() == 0:
                continue
            gt_all = out['gt_disp'].cpu().numpy() if isinstance(out['gt_disp'], torch.Tensor) else out['gt_disp']
            # Sample node-wise GT as before (you already compute node_gts via numpy segments). When building gt_nodes:
            gt_nodes = torch.tensor(node_gts[mask_valid], dtype=torch.float32, device=device)

            pred_valid = pred_nodes[mask_valid]           # this keeps requires_grad=True
            loss_l1 = F.l1_loss(pred_valid, gt_nodes)
            
            # backward
            opt.zero_grad()
            loss_l1.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            total_loss += float(loss_l1.item())
            count += 1

        epoch_loss = total_loss / max(1, count)
        print(f"🧩 Epoch {epoch:02d}: Train L1 = {epoch_loss:.4f}")

        # save checkpoint if improved
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), f"stereo_graphnet_best.pth")
            print("💾 Saved best model checkpoint!")

        # optional: visualize every few epochs
        # optional: visualize every few epochs
        # optional: visualize every few epochs
        if epoch % 5 == 0:
            with torch.no_grad():
                sample = ds[0]
                res = model.forward_from_paths(*sample)
                disp_tensor = res['pixel_disp_refined']
                if isinstance(disp_tensor, torch.Tensor):
                    disp_np = disp_tensor.detach().cpu().numpy()
                else:
                    disp_np = disp_tensor
                disp_vis = cv2.normalize(disp_np, None, 0, 255, cv2.NORM_MINMAX)
                cv2.imwrite(f"disp_refined_epoch{epoch}.png", np.uint8(disp_vis))
                print(f"🖼️ Saved visualization for epoch {epoch}")




if __name__ == "__main__":
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ✅ Your dataset root folders
    TRAIN_ROOT = r"D:\WHU_stereo_dataset\train\012_98"
    VAL_ROOT   = r"D:\WHU_stereo_dataset\train\012_95"

    print(f"🚀 Starting training on {device} ...")
    train_loop(TRAIN_ROOT, VAL_ROOT, device)



# # -------------------- Quick inference helper --------------------
# if __name__ == '__main__':
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     # Please edit these paths
#     TRAIN_ROOT = r"D:\WHU_stereo_dataset\train"
#     VAL_ROOT = r"D:\WHU_stereo_dataset\val"

#     # train_loop(TRAIN_ROOT, VAL_ROOT, device=device)
#     # Quick demo inference (you can run the model on single pair):
#     model = StereoGraphNet(dmax=128, node_feat_dim=128, device=device).to(device)
#     # load pretrained weights if you have
#     # model.load_state_dict(torch.load('mystereo_weights.pth'))
#     left_path = 'data/left/000001.png'
#     right_path = 'data/right/000001.png'
#     disp = 'data/disparity/000001.png'
#     if os.path.isfile(left_path) and os.path.isfile(right_path) and os.path.isfile(disp):
#         res = model.forward_from_paths(left_path, right_path, disp)
#         # save visualization
#         pd = res['pixel_disp_refined']
#         outv = (pd / (pd.max() + 1e-6) * 255).astype(np.uint8)
#         cv2.imwrite('disp_refined.png', outv)
#     else:
#         print('Edit paths in the script to run demo or call train_loop()')
