import argparse
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from lightglue import SuperPoint, LightGlue, viz2d

try:
    from romatch import roma_outdoor
except Exception:
    roma_outdoor = None

try:
    from src.loftr import LoFTR, full_default_cfg, reparameter
    from copy import deepcopy
except Exception:
    LoFTR = None

try:
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
except Exception:
    VGGT = None


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def tensor_from_image(img):
    t = torch.from_numpy(img).float() / 255.0
    if t.ndim == 3:
        t = t.permute(2, 0, 1)  # CHW
    return t


def match_lightglue(img0, img1, device):
    sp = SuperPoint(max_num_keypoints=2048).to(device)
    matcher = LightGlue(features="superpoint").to(device)
    matcher.eval()

    t0 = tensor_from_image(img0)[None].to(device)
    t1 = tensor_from_image(img1)[None].to(device)

    with torch.no_grad():
        feats0 = sp.extract(t0)
        feats1 = sp.extract(t1)
        matches = matcher({"image0": feats0, "image1": feats1})

    # LightGlue returns a dict with a list of matched indices per image pair
    # matches["matches"][0] has shape [S x 2] for the first (and only) batch
    mkpts0_idx = matches["matches"][0][:, 0]
    mkpts1_idx = matches["matches"][0][:, 1]
    mkpts0 = feats0["keypoints"][0, mkpts0_idx].cpu().numpy()
    mkpts1 = feats1["keypoints"][0, mkpts1_idx].cpu().numpy()
    return mkpts0, mkpts1


def match_eloftr(img0, img1, device):
    if LoFTR is None:
        raise ImportError("EfficientLoFTR is not available")
    config = deepcopy(full_default_cfg)
    matcher = LoFTR(config)
    matcher = reparameter(matcher).to(device)
    matcher.eval()

    img0_t = torch.from_numpy(cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY))[None, None].to(device) / 255.0
    img1_t = torch.from_numpy(cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY))[None, None].to(device) / 255.0
    batch = {"image0": img0_t, "image1": img1_t}
    with torch.no_grad():
        matcher(batch)
    mkpts0 = batch["mkpts0_f"].cpu().numpy()
    mkpts1 = batch["mkpts1_f"].cpu().numpy()
    return mkpts0, mkpts1


def match_vggt(img0_path, img1_path, device):
    if VGGT is None:
        raise ImportError("VGGT package not available")
    images = load_and_preprocess_images([img0_path, img1_path])
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(
        torch.hub.load_state_dict_from_url(_URL, map_location=device)
    )
    model = model.to(device).half().eval()
    images = images.to(device).half()[None]

    H, W = images.shape[-2:]
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    query = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)

    with torch.no_grad(), torch.cuda.amp.autocast():
        agg, ps_idx = model.aggregator(images)

        # Process query points in smaller batches to reduce memory usage
        tracks = []
        chunk = 8192
        for i in range(0, query.shape[0], chunk):
            q = query[i : i + chunk]
            t, _, _ = model.track_head(
                agg, images, ps_idx, query_points=q[None]
            )
            tracks.append(t[-1])

        track = torch.cat(tracks, dim=2)

    mkpts0 = query.cpu().numpy()
    mkpts1 = track[0, 1].cpu().numpy()
    return mkpts0, mkpts1


def match_roma(img0_path, img1_path, device):
    if roma_outdoor is None:
        raise ImportError("RoMa package not available")
    model = roma_outdoor(device=device)
    warp, certainty = model.match(img0_path, img1_path, device=device)
    Ht, W2, _ = warp.shape
    W = W2 // 2
    grid_y, grid_x = torch.meshgrid(torch.linspace(-1 + 1 / Ht, 1 - 1 / Ht, Ht, device=device),
                                    torch.linspace(-1 + 1 / W, 1 - 1 / W, W, device=device), indexing="ij")
    grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
    mkpts0 = ((grid + 1) * torch.tensor([(W - 1) / 2, (Ht - 1) / 2], device=device)).cpu().numpy()
    mkpts1 = warp[:,:W,2:].reshape(-1,2).cpu().numpy()
    return mkpts0, mkpts1


def main(img0_path, img1_path, method="all", output=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img0 = load_image(img0_path)
    img1 = load_image(img1_path)

    methods = ["lightglue", "vggt", "roma", "efficientloftr"] if method == "all" else [method]
    results = {}

    if "lightglue" in methods:
        mkpts0, mkpts1 = match_lightglue(img0, img1, device)
        results["lightglue"] = (mkpts0, mkpts1)

    if "efficientloftr" in methods:
        mkpts0, mkpts1 = match_eloftr(img0, img1, device)
        results["efficientloftr"] = (mkpts0, mkpts1)

    if "vggt" in methods:
        mkpts0, mkpts1 = match_vggt(img0_path, img1_path, device)
        results["vggt"] = (mkpts0, mkpts1)

    if "roma" in methods:
        mkpts0, mkpts1 = match_roma(img0_path, img1_path, device)
        results["roma"] = (mkpts0, mkpts1)

    img_h, img_w = img0.shape[:2]
    for name, (k0, k1) in results.items():
        H, _ = cv2.findHomography(k1, k0, cv2.RANSAC)
        warped = cv2.warpPerspective(img1, H, (img_w, img_h))

        viz2d.plot_images([img0, img1])
        viz2d.plot_matches(k0, k1)
        plt.tight_layout()
        if output:
            plt.savefig(f"{name}_matches.png")
        else:
            plt.show()

        plt.figure()
        plt.title(f"Warped with {name}")
        plt.imshow(warped)
        plt.axis("off")
        if output:
            plt.savefig(f"{name}_warp.png")
        else:
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image pair matching demo")
    parser.add_argument("img0")
    parser.add_argument("img1")
    parser.add_argument("--method", default="all", choices=["all", "lightglue", "vggt", "roma", "efficientloftr"], help="Matching backend")
    parser.add_argument("--output", default=None, help="Optional output figure path prefix")
    args = parser.parse_args()
    main(args.img0, args.img1, args.method, args.output)
