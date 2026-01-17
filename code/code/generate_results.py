import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =========================
# CONFIG
# =========================
DATA_ROOT = "/content/drive/MyDrive/ImageProcProject/data_raw"
OUT_DIR = "../results"  # relative to /code when run from code/
IMG_SIZE = (64, 64)
EVERY_N = 1
BLOCK_SIZE = 5
MOTION_THR = 15.0
MAX_BLOCKS_PER_VIDEO = 2500

# baseline and improved (tau)
TAU_BASELINE = 1.0
TAU_IMPROVED = 5.0

# colors (BGR for OpenCV)
COLOR_WAVE = (255, 0, 255)  # magenta
COLOR_WALK = (0, 255, 255)  # yellow

SEED = 42


# =========================
# IO / PREPROCESS
# =========================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def list_videos(split):
    paths, labels = [], []
    for cls in ["wave", "walk"]:
        vp = sorted(glob.glob(os.path.join(DATA_ROOT, split, cls, "*")))
        paths += vp
        labels += ([0]*len(vp) if cls == "wave" else [1]*len(vp))
    return paths, np.array(labels, dtype=int)

def load_video_frames(video_path, size=(64,64), every_n=1):
    cap = cv2.VideoCapture(video_path)
    frames = []
    i = 0
    ok = True
    while ok:
        ok, frame = cap.read()
        if not ok:
            break
        if i % every_n == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            frames.append(frame.astype(np.float32))
        i += 1
    cap.release()
    if len(frames) == 0:
        return None
    return np.stack(frames, axis=0)  # (T, H, W)

# =========================
# BLOCKS + FEATURES (DCT)
# =========================
def sample_spatiotemporal_blocks(video, block_size=5, motion_thr=15.0, max_blocks=2500, seed=42):
    """
    video: (T,H,W) float32
    Returns blocks: (N, bs, bs, bs) in (t, y, x) cube
    """
    rng = np.random.default_rng(seed)
    T, H, W = video.shape
    bs = block_size

    if T < bs or H < bs or W < bs:
        return np.zeros((0, bs, bs, bs), dtype=np.float32), None

    # quick motion map: std over time at each pixel
    std_map = video.std(axis=0)  # (H,W)
    motion_mask = std_map >= motion_thr

    # candidate top-left corners (y,x) where motion exists in the spatial patch
    ys, xs = np.where(motion_mask)
    if len(ys) == 0:
        return np.zeros((0, bs, bs, bs), dtype=np.float32), None

    # sample random block centers from motion pixels
    idx = rng.choice(len(ys), size=min(max_blocks*2, len(ys)), replace=False)
    centers = list(zip(ys[idx], xs[idx]))

    blocks = []
    coords = []  # (t0,y0,x0)
    tries = 0
    while len(blocks) < max_blocks and tries < len(centers)*3:
        cy, cx = centers[tries % len(centers)]
        # choose random start t
        t0 = rng.integers(0, T - bs + 1)

        # convert center to top-left
        y0 = int(np.clip(cy - bs//2, 0, H - bs))
        x0 = int(np.clip(cx - bs//2, 0, W - bs))

        cube = video[t0:t0+bs, y0:y0+bs, x0:x0+bs]  # (bs,bs,bs)
        # require motion inside cube (std in cube)
        if cube.std() >= motion_thr:
            blocks.append(cube)
            coords.append((t0, y0, x0))
        tries += 1

    if len(blocks) == 0:
        return np.zeros((0, bs, bs, bs), dtype=np.float32), None

    return np.stack(blocks, axis=0).astype(np.float32), np.array(coords, dtype=int)

def dct3(block):
    # block shape: (bs,bs,bs) float
    # apply 1D DCT along each axis
    from scipy.fftpack import dct
    x = dct(block, axis=0, norm="ortho")
    x = dct(x, axis=1, norm="ortho")
    x = dct(x, axis=2, norm="ortho")
    return x

def block_to_dct_vector(block):
    coeff = np.abs(dct3(block))
    return coeff.flatten()

# =========================
# FEATURE SELECTION + BINARIZATION
# =========================
def select_thresholds_and_topk(X, y, num_bins=30, topk=30):
    """
    Simple MI-like proxy using class-conditional separation on binned thresholds.
    Returns:
      feat_idx: indices of topk features
      feat_thr: threshold per selected feature
    """
    rng = np.random.default_rng(0)
    n, d = X.shape

    # compute candidate thresholds by percentiles
    thr = np.percentile(X, np.linspace(10, 90, num_bins), axis=0)  # (num_bins, d)

    # score each feature by best separation over thresholds
    # proxy: |P(x>t|y=1) - P(x>t|y=0)|
    y0 = (y == 0)
    y1 = (y == 1)

    best_score = np.zeros(d, dtype=np.float32)
    best_thr = np.zeros(d, dtype=np.float32)

    for j in range(d):
        bj = -1.0
        bt = thr[0, j]
        xj = X[:, j]
        for t in thr[:, j]:
            p1 = (xj[y1] > t).mean() if y1.any() else 0.0
            p0 = (xj[y0] > t).mean() if y0.any() else 0.0
            s = abs(p1 - p0)
            if s > bj:
                bj = s
                bt = t
        best_score[j] = bj
        best_thr[j] = bt

    feat_idx = np.argsort(-best_score)[:topk]
    feat_thr = best_thr[feat_idx]
    return feat_idx, feat_thr, best_score[feat_idx]

def binarize_selected(X, feat_idx, feat_thr):
    Xs = X[:, feat_idx]
    return (Xs > feat_thr.reshape(1, -1)).astype(np.int8)

# =========================
# DATASET COLLECTION
# =========================
def collect_from_paths(paths, labels, block_size=5, motion_thr=15.0, max_blocks_per_video=2500, seed=42):
    X_list, y_list, meta = [], [], []
    for vp, lab in zip(paths, labels):
        vid = load_video_frames(vp, size=IMG_SIZE, every_n=EVERY_N)
        if vid is None:
            continue
        blocks, coords = sample_spatiotemporal_blocks(
            vid, block_size=block_size, motion_thr=motion_thr,
            max_blocks=max_blocks_per_video, seed=seed
        )
        if len(blocks) == 0:
            continue

        Xc = np.stack([block_to_dct_vector(b) for b in blocks], axis=0)
        yc = np.full((Xc.shape[0],), lab, dtype=np.int32)

        X_list.append(Xc)
        y_list.append(yc)

        # store metadata for visualization
        for c in coords:
            meta.append((vp, int(lab), int(c[0]), int(c[1]), int(c[2])))

    if len(X_list) == 0:
        return np.zeros((0, block_size**3), dtype=np.float32), np.zeros((0,), dtype=np.int32), []

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y, meta

# =========================
# CONFIDENCE FILTERING (TAU)
# =========================
def predict_with_tau(model, X_bin, tau=1.0):
    proba = model.predict_proba(X_bin)  # (N,2)
    p0 = proba[:, 0]
    p1 = proba[:, 1]
    pred = (p1 >= p0).astype(int)

    win = np.maximum(p0, p1)
    lose = np.minimum(p0, p1) + 1e-12
    ratio = win / lose
    keep = ratio >= tau
    return pred, keep, proba

# =========================
# VISUALIZATION (OVERLAY)
# =========================
def overlay_blocks_on_frame(frame_gray, blocks_meta, blocks_pred, blocks_keep, tau, alpha=0.55):
    """
    frame_gray: (H,W) uint8
    blocks_meta: list of tuples (video_path, ytrue, t0, y0, x0)
    blocks_pred: array int (0 wave, 1 walk)
    blocks_keep: boolean mask
    """
    h, w = frame_gray.shape
    frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

    overlay = frame_bgr.copy()

    for (vp, ytrue, t0, y0, x0), pr, kp in zip(blocks_meta, blocks_pred, blocks_keep):
        if not kp:
            continue
        color = COLOR_WAVE if pr == 0 else COLOR_WALK
        cv2.rectangle(overlay, (x0, y0), (x0 + BLOCK_SIZE, y0 + BLOCK_SIZE), color, thickness=-1)

    out = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)
    return out

def save_confusion_matrix(cm, out_path, class_names=("wave", "walk"), title="Confusion Matrix"):
    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0,1], class_names)
    plt.yticks([0,1], class_names)
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_accuracy_vs_tau(tau_list, acc_list, cov_list, out_path):
    plt.figure()
    plt.plot(tau_list, acc_list, marker="o")
    plt.xlabel("tau (confidence ratio)")
    plt.ylabel("Accuracy (classified blocks)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    # coverage plot
    cov_path = out_path.replace("accuracy_vs_tau", "coverage_vs_tau")
    plt.figure()
    plt.plot(tau_list, cov_list, marker="o")
    plt.xlabel("tau (confidence ratio)")
    plt.ylabel("Coverage (fraction classified)")
    plt.tight_layout()
    plt.savefig(cov_path, dpi=200)
    plt.close()

# =========================
# MAIN
# =========================
def main():
    ensure_dir(os.path.abspath(os.path.join(os.path.dirname(__file__), OUT_DIR)))

    # Train
    train_paths, train_labels = list_videos("train")
    Xtr_cont, ytr, _ = collect_from_paths(
        train_paths, train_labels,
        block_size=BLOCK_SIZE, motion_thr=MOTION_THR,
        max_blocks_per_video=MAX_BLOCKS_PER_VIDEO, seed=SEED
    )

    feat_idx, feat_thr, _ = select_thresholds_and_topk(Xtr_cont, ytr, num_bins=30, topk=30)
    Xtr_bin = binarize_selected(Xtr_cont, feat_idx, feat_thr)

    model = BernoulliNB(alpha=1.0)
    model.fit(Xtr_bin, ytr)

    # Test + metrics
    test_paths, test_labels = list_videos("test")
    Xte_cont, yte, meta = collect_from_paths(
        test_paths, test_labels,
        block_size=BLOCK_SIZE, motion_thr=MOTION_THR,
        max_blocks_per_video=MAX_BLOCKS_PER_VIDEO, seed=SEED+1
    )
    Xte_bin = binarize_selected(Xte_cont, feat_idx, feat_thr)

    # baseline tau=1 (all blocks)
    pred_base, keep_base, _ = predict_with_tau(model, Xte_bin, tau=TAU_BASELINE)
    acc_base = accuracy_score(yte[keep_base], pred_base[keep_base])
    cm_base = confusion_matrix(yte[keep_base], pred_base[keep_base])

    # improved tau=5
    pred_imp, keep_imp, _ = predict_with_tau(model, Xte_bin, tau=TAU_IMPROVED)
    acc_imp = accuracy_score(yte[keep_imp], pred_imp[keep_imp])
    cm_imp = confusion_matrix(yte[keep_imp], pred_imp[keep_imp])

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), OUT_DIR))

    save_confusion_matrix(cm_base, os.path.join(out_dir, "confusion_matrix_baseline.png"),
                          title=f"Confusion Matrix (tau={TAU_BASELINE})")
    save_confusion_matrix(cm_imp, os.path.join(out_dir, "confusion_matrix_improved.png"),
                          title=f"Confusion Matrix (tau={TAU_IMPROVED})")

    # tau sweep plot
    tau_list = [1.0, 1.2, 1.5, 2.0, 3.0, 5.0]
    acc_list, cov_list = [], []
    for t in tau_list:
        pred, keep, _ = predict_with_tau(model, Xte_bin, tau=t)
        if keep.sum() == 0:
            acc_list.append(np.nan)
            cov_list.append(0.0)
        else:
            acc_list.append(accuracy_score(yte[keep], pred[keep]))
            cov_list.append(float(keep.mean()))
    save_accuracy_vs_tau(tau_list, acc_list, cov_list, os.path.join(out_dir, "accuracy_vs_tau.png"))

    # print summary (for report)
    print("=== BASELINE ===")
    print(f"tau={TAU_BASELINE}, acc={acc_base:.4f}, coverage={keep_base.mean():.3f}, blocks={keep_base.sum()}")
    print(classification_report(yte[keep_base], pred_base[keep_base], target_names=["wave(0)", "walk(1)"]))
    print(cm_base)

    print("\n=== IMPROVED ===")
    print(f"tau={TAU_IMPROVED}, acc={acc_imp:.4f}, coverage={keep_imp.mean():.3f}, blocks={keep_imp.sum()}")
    print(classification_report(yte[keep_imp], pred_imp[keep_imp], target_names=["wave(0)", "walk(1)"]))
    print(cm_imp)

    # -------------------------
    # OVERLAYS: pick 1 walk + 1 wave test video, save baseline/improved frames
    # -------------------------
    # select one test video per class
    wave_videos = [p for p in test_paths if os.path.basename(os.path.dirname(p)) == "wave"]
    walk_videos = [p for p in test_paths if os.path.basename(os.path.dirname(p)) == "walk"]
    pick_wave = wave_videos[0] if len(wave_videos) else None
    pick_walk = walk_videos[0] if len(walk_videos) else None

    def save_overlays_for_video(video_path, label_name):
        vid = load_video_frames(video_path, size=IMG_SIZE, every_n=EVERY_N)
        if vid is None:
            return

        blocks, coords = sample_spatiotemporal_blocks(
            vid, block_size=BLOCK_SIZE, motion_thr=MOTION_THR,
            max_blocks=MAX_BLOCKS_PER_VIDEO, seed=SEED+9
        )
        if len(blocks) == 0:
            return

        Xc = np.stack([block_to_dct_vector(b) for b in blocks], axis=0)
        Xb = binarize_selected(Xc, feat_idx, feat_thr)

        pred_b, keep_b, _ = predict_with_tau(model, Xb, tau=TAU_BASELINE)
        pred_i, keep_i, _ = predict_with_tau(model, Xb, tau=TAU_IMPROVED)

        # choose a representative frame (middle)
        t_show = int(vid.shape[0] // 2)
        frame = np.clip(vid[t_show], 0, 255).astype(np.uint8)

        meta_local = [(video_path, -1, int(c[0]), int(c[1]), int(c[2])) for c in coords]

        out_base = overlay_blocks_on_frame(frame, meta_local, pred_b, keep_b, TAU_BASELINE)
        out_imp  = overlay_blocks_on_frame(frame, meta_local, pred_i, keep_i, TAU_IMPROVED)

        cv2.imwrite(os.path.join(out_dir, f"overlay_baseline_{label_name}.png"), out_base)
        cv2.imwrite(os.path.join(out_dir, f"overlay_improved_{label_name}.png"), out_imp)

    if pick_wave:
        save_overlays_for_video(pick_wave, "wave")
    if pick_walk:
        save_overlays_for_video(pick_walk, "walk")

    print("\nSaved outputs to:", out_dir)

if __name__ == "__main__":
    main()

ֿ
