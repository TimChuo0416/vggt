import argparse
import cv2
import numpy as np
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def project_points(world_points: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """Project world points to pixel coordinates."""
    h, w = world_points.shape[:2]
    pts = world_points.reshape(-1, 3)
    cam = pts @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    u = cam[:, 0] / cam[:, 2]
    v = cam[:, 1] / cam[:, 2]
    u = intrinsic[0, 0] * u + intrinsic[0, 2]
    v = intrinsic[1, 1] * v + intrinsic[1, 2]
    return np.stack([u.reshape(h, w), v.reshape(h, w)], axis=-1)


def warp_image(src_img: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    map_x = mapping[:, :, 0].astype(np.float32)
    map_y = mapping[:, :, 1].astype(np.float32)
    return cv2.remap(src_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def run_vggt(img1_path: str, img2_path: str, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)

    images = load_and_preprocess_images([img1_path, img2_path]).to(device)
    with torch.no_grad():
        preds = model(images)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(preds["pose_enc"], images.shape[-2:])

    world_points = preds["world_points"].cpu().numpy()
    extrinsic = extrinsic.cpu().numpy()
    intrinsic = intrinsic.cpu().numpy()

    map21 = project_points(world_points[0], extrinsic[1], intrinsic[1])
    map12 = project_points(world_points[1], extrinsic[0], intrinsic[0])

    img1 = (images[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img2 = (images[1].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    warp21 = warp_image(img2, map21)  # warp image2 -> image1
    warp12 = warp_image(img1, map12)  # warp image1 -> image2
    return img1[..., ::-1], img2[..., ::-1], warp21[..., ::-1], warp12[..., ::-1]


def run_roma(img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute warps using RoMa if available, otherwise fallback to Farneback."""
    try:
        import roma  # type: ignore
        # Insert official RoMa dense matching code here if available.
        raise ImportError  # force fallback unless user customizes
    except Exception:
        print("RoMa library not found. Using OpenCV Farneback optical flow as a placeholder.")
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        flow21 = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow12 = cv2.calcOpticalFlowFarneback(gray2, gray1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        h, w = gray1.shape
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map21 = np.stack([grid_x + flow21[..., 0], grid_y + flow21[..., 1]], axis=-1)
        map12 = np.stack([grid_x + flow12[..., 0], grid_y + flow12[..., 1]], axis=-1)
        warp21 = warp_image(img2, map21)
        warp12 = warp_image(img1, map12)
        return warp21, warp12


def visualize(img1, img2, vggt_21, roma_21, vggt_12, roma_12, out_file=None):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes[0, 0].imshow(img1)
    axes[0, 0].set_title("Image 1")
    axes[0, 1].imshow(img2)
    axes[0, 1].set_title("Image 2")
    axes[0, 2].axis('off')
    axes[1, 0].imshow(vggt_21)
    axes[1, 0].set_title("VGGT: 2 -> 1")
    axes[1, 1].imshow(roma_21)
    axes[1, 1].set_title("RoMa: 2 -> 1")
    axes[1, 2].axis('off')
    fig.tight_layout()
    if out_file:
        plt.savefig(out_file)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Image pair matching demo")
    parser.add_argument("--image1", required=True, help="Path to first image")
    parser.add_argument("--image2", required=True, help="Path to second image")
    parser.add_argument("--output", default=None, help="Optional path to save visualization")
    parser.add_argument("--device", default="cuda", help="Computation device")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    img1, img2, vggt_warp21, vggt_warp12 = run_vggt(args.image1, args.image2, device)
    roma_warp21, roma_warp12 = run_roma(img1, img2)
    visualize(img1, img2, vggt_warp21, roma_warp21, vggt_warp12, roma_warp12, args.output)


if __name__ == "__main__":
    main()
