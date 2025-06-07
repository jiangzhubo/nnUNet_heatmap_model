import os
import numpy as np
import pickle
from scipy.ndimage import gaussian_filter


def make_case(idx: int, out_dir: str, shape=(32, 32)):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"case_{idx:04d}")

    seg = np.zeros(shape, dtype=np.int16)
    center = np.random.randint(8, shape[0] - 8, size=2)
    radius = 5
    for x in range(shape[0]):
        for y in range(shape[1]):
            if (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2:
                seg[x, y] = 1

    img = seg.astype(np.float32) + 0.1 * np.random.randn(*shape).astype(np.float32)
    heatmap = gaussian_filter(seg.astype(np.float32), sigma=2)

    class_locations = {
        0: np.argwhere(seg == 0).tolist(),
        1: np.argwhere(seg == 1).tolist(),
    }
    properties = {"class_locations": class_locations}

    # the nnUNetDataset class expects .npz files for inferring case identifiers
    # We therefore store image and segmentation in a compressed npz archive.
    np.savez_compressed(base + ".npz", data=img[None], seg=seg[None])
    # heatmaps are optional and loaded separately
    np.save(base + "_heatmap.npy", heatmap)
    with open(base + ".pkl", "wb") as f:
        pickle.dump(properties, f)


def generate_dataset(out_dir: str, num_cases: int = 4):
    np.random.seed(0)
    for i in range(num_cases):
        make_case(i, out_dir)


if __name__ == "__main__":
    generate_dataset(os.path.join(os.path.dirname(__file__), "dummy_heatmap_dataset"))
