# Shared helpers: data loading, model loading, the sliding-window encoder,
# the kNN decoder, and the position -> LED color mapping.
import colorsys
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET = "rat-hippocampus-single-achilles"
BIN_S = 0.025        # one time step = 25 ms (same bin size as the CEBRA demo)
SPLIT = 0.8          # first 80% train, last 20% test (same as CEBRA_Reproduce_Workflow)
N_NEIGHBORS = 36     # same kNN decoder as the notebook


def load_data():
    """Returns neural (T, 120) float32, labels (T, 3) [position_m, right, left], split index."""
    os.environ.setdefault("CEBRA_DATADIR", str(ROOT / "data"))
    import cebra.datasets
    ds = cebra.datasets.init(DATASET)
    neural = ds.neural.numpy().astype(np.float32)
    label = ds.continuous_index.numpy()
    return neural, label, int(SPLIT * len(neural))


def shuffled_labels(label, seed=0):
    # same permutation demo_hippocampus.py used for the shuffled control
    return np.random.default_rng(seed).permutation(label)


def load_model(path):
    """Loads a saved CEBRA model onto the CPU and returns its torch network."""
    import torch
    from cebra.integrations.sklearn import cebra as cebra_sklearn
    # CEBRA.load() moves the model back to the device it was trained on ("cuda" on ICE),
    # which crashes on a laptop. So load it ourselves and point it at the CPU first.
    # weights_only=False is fine here: these are our own checkpoints.
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    ckpt["state"]["device_"] = "cpu"
    ckpt["args"]["device"] = "cpu"
    m = cebra_sklearn._load_cebra_with_sklearn_backend(ckpt)
    return m.model_.eval()


class WindowEncoder:
    """Runs CEBRA's encoder on windows of `left + right` samples (10 for offset10-model).

    Output index i comes from the window X[i - left : i + right].
    So a window that ENDS at sample t gives the embedding for i = t - (right - 1).
    """

    def __init__(self, net):
        import torch
        self.torch, self.net = torch, net
        offset = net.get_offset()
        self.left, self.right = offset.left, offset.right
        self.width = offset.left + offset.right

    def embed_window(self, window):
        """window: (width, neurons) -> (dim,). The real-time path: one call per sample."""
        x = self.torch.from_numpy(np.ascontiguousarray(window.T)[None])
        with self.torch.no_grad():
            return self.net(x).reshape(-1).numpy()

    def embed_all(self, X, batch=1024):
        """Offline path: embed every sample of X at once (edges padded by repeating)."""
        Xp = np.concatenate([np.repeat(X[:1], self.left, 0), X, np.repeat(X[-1:], self.right - 1, 0)])
        windows = self.torch.from_numpy(np.ascontiguousarray(Xp.T)).unfold(1, self.width, 1)  # (C, T, w)
        windows = windows.permute(1, 0, 2)  # (T, C, w)
        out = []
        with self.torch.no_grad():
            for s in range(0, len(windows), batch):
                b = windows[s:s + batch].contiguous()
                out.append(self.net(b).reshape(len(b), -1).numpy())
        return np.concatenate(out)


class Decoder:
    """kNN position + direction decoder (same settings as the notebook)."""

    def __init__(self, emb_train, lbl_train):
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        self.pos = KNeighborsRegressor(N_NEIGHBORS, metric="cosine").fit(emb_train, lbl_train[:, 0])
        self.dir = KNeighborsClassifier(N_NEIGHBORS, metric="cosine").fit(emb_train, lbl_train[:, 1])

    def predict(self, emb):
        """emb: (n, dim) -> (positions (n,), right_running (n,))"""
        return self.pos.predict(emb), self.dir.predict(emb)


class ColorMap:
    """Position on the track -> hue (blue at one end, red at the other).
    Direction -> brightness (running right = full, running left = dim)."""

    def __init__(self, positions, dim_level=0.3):
        self.pmin, self.pmax = float(np.min(positions)), float(np.max(positions))
        self.dim_level = dim_level

    def rgb(self, pos, right_running):
        frac = min(max((pos - self.pmin) / (self.pmax - self.pmin), 0.0), 1.0)
        hue = (2 / 3) * (1 - frac)  # 2/3 = blue, 0 = red
        value = 1.0 if right_running >= 0.5 else self.dim_level
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, value)
        return int(r * 255), int(g * 255), int(b * 255)


def ansi_block(rgb, width=6):
    r, g, b = rgb
    return f"\x1b[48;2;{r};{g};{b}m{' ' * width}\x1b[0m"
