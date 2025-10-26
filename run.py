# // 22CSB0C20 CHANDU CHERUPALLY
"""
run.py - Hybrid low-res face matcher 

Purpose:
- Evaluate strategies for matching a high-resolution enrolled gallery image
  to low-resolution probe images (CCTV-like).
- Demonstrates that direct comparison (Naive) fails and that upscaling (Bicubic)
  improves identification.

Naive (strict) behavior:
- Downscales the probe to low_res and upscales back to embed_size using
  nearest-neighbor; no fallback to higher-quality embedding. This preserves
  blockiness and demonstrates the failure of direct comparison.

Bicubic Probe behavior:
- Downscales the probe to low_res then upscales using bicubic interpolation
  to embed_size; bicubic smooths and restores gradients so embeddings are
  more stable and matching improves.
"""

import os
import random
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
import face_recognition
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import csv
import datetime

# ------------------------
# Utilities
# ------------------------

def list_identities(dataset_dir, min_imgs=2):
    """Return list of (identity_name, [image_paths]) for folders with >= min_imgs."""
    dataset_dir = Path(dataset_dir)
    identities = []
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")
    for sub in sorted(os.listdir(dataset_dir)):
        p = dataset_dir / sub
        if p.is_dir():
            imgs = [str(p / f) for f in sorted(os.listdir(p)) if f.lower().endswith(('.jpg','.jpeg','.png'))]
            if len(imgs) >= min_imgs:
                identities.append((sub, imgs))
    return identities

def safe_load_image(path):
    """Load image with PIL and return RGB uint8 array or None on failure."""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            return np.array(im)
    except Exception:
        return None

def detect_and_crop_face(img_bgr, expand=0.25, img_path=None):
    """Detect largest face and return expanded crop in BGR; uses PIL fallback if needed."""
    if img_bgr is None:
        return None

    # Ensure image is uint8 3-channel BGR before detection
    if img_bgr.dtype != np.uint8:
        if np.issubdtype(img_bgr.dtype, np.floating):
            img_bgr = (np.clip(img_bgr, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            img_bgr = img_bgr.astype(np.uint8)

    # Convert gray/alpha to BGR
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    elif img_bgr.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

    # Convert to RGB for face_recognition detection
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Try face_locations; fallback to PIL-based array if necessary
    try:
        locs = face_recognition.face_locations(rgb, model='hog')
    except Exception:
        if img_path:
            pil_rgb = safe_load_image(img_path)
            if pil_rgb is not None:
                try:
                    locs = face_recognition.face_locations(pil_rgb, model='hog')
                except Exception:
                    return None
            else:
                return None
        else:
            return None

    if not locs:
        return None

    # Choose the largest detected face (most robust for group photos)
    def area(box):
        top, right, bottom, left = box
        return (bottom - top) * (right - left)
    locs_sorted = sorted(locs, key=area, reverse=True)
    top, right, bottom, left = locs_sorted[0]

    # Expand bounding box slightly and crop from original BGR image
    h = bottom - top
    w = right - left
    top = max(0, int(top - expand * h))
    bottom = min(img_bgr.shape[0], int(bottom + expand * h))
    left = max(0, int(left - expand * w))
    right = min(img_bgr.shape[1], int(right + expand * w))

    crop = img_bgr[top:bottom, left:right]
    if crop.size == 0:
        return None
    return crop

def resize_keep_aspect(img, target_size=(160,160)):
    """Resize image to target_size using bicubic interpolation (used for embedding)."""
    return cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)

def downsample_then_upscale(img, low_res=(24,24), upscale_to=(160,160)):
    """Downsample with AREA then upscale with CUBIC to simulate degradation+restore."""
    small = cv2.resize(img, low_res, interpolation=cv2.INTER_AREA)
    up = cv2.resize(small, upscale_to, interpolation=cv2.INTER_CUBIC)
    return up

def make_dnn_sr(sr_model_path, sr_name='edsr', scale=4):
    """Load OpenCV's DNN Super-Resolution model if available (opencv-contrib required)."""
    if not hasattr(cv2, 'dnn_superres'):
        print("OpenCV dnn_superres not available. Install opencv-contrib-python to use SR.")
        return None
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(sr_model_path)
        sr.setModel(sr_name, scale)
        return sr
    except Exception as e:
        print("Failed to load DNN SR model:", e)
        return None

def sr_upscale_with_model(sr, img):
    """Upscale using loaded DNN SR model; fallback to original on error."""
    if sr is None:
        return img
    try:
        return sr.upsample(img)
    except Exception:
        return img

def get_embedding_for_face(face_bgr):
    """Return 128-d face embedding or None if face_recognition can't encode."""
    if face_bgr is None or face_bgr.size == 0:
        return None
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    encs = face_recognition.face_encodings(rgb)
    if not encs:
        return None
    return encs[0]

# ------------------------
# Main pipeline
# ------------------------

def build_gallery_and_probes(dataset_dir, max_identities=50, seed=42, dump_ids=None):
    """Select deterministic subset of identities, write chosen IDs if requested."""
    ids = list_identities(dataset_dir, min_imgs=2)
    if not ids:
        raise RuntimeError("No identities found in dataset_dir")
    rng = random.Random(seed)
    rng.shuffle(ids)
    selected = ids[:max_identities]

    # Optionally write chosen identity names for reproducibility across runs
    if dump_ids:
        try:
            with open(dump_ids, 'w') as f:
                for name, imgs in selected:
                    f.write(name + '\n')
        except Exception as e:
            print("Failed to write dump_ids:", e)

    # Gallery stores full list of image paths per identity for fallback attempts
    gallery = [(name, imgs) for name, imgs in selected]

    # Probes are remaining images (except the one used for gallery)
    probes = []
    for name, imgs in selected:
        imgs_sh = imgs[:]
        rng.shuffle(imgs_sh)
        for p in imgs_sh[1:]:
            probes.append((name, p))
    return gallery, probes

def process_and_embed(gallery_list, probes, sr=None, method='naive', low_res=(24,24), embed_size=(160,160),
                      strict_naive=False, save_examples=False, examples_dir="examples", max_examples=8):
    """
    Process gallery and probes; return dictionary of gallery embeddings and list of probe tuples.
    strict_naive enforces truly low-res probes (nearest-neighbor upscaling) and skips fallback.
    """
    import os
    gallery_embs = {}
    gallery_samples = {}
    # Prepare examples directory if requested
    if save_examples and not os.path.exists(examples_dir):
        os.makedirs(examples_dir, exist_ok=True)

    # Build gallery embeddings: try multiple images per identity until one yields an embedding
    for name, imgs in tqdm(gallery_list, desc="Gallery (identities)"):
        emb = None
        chosen_proc = None
        chosen_path = None
        for path in imgs:
            img = cv2.imread(path)
            if img is None:
                # PIL fallback for exotic formats
                pil_rgb = safe_load_image(path)
                if pil_rgb is None:
                    continue
                img = cv2.cvtColor(pil_rgb, cv2.COLOR_RGB2BGR)
            crop = detect_and_crop_face(img, img_path=path)
            if crop is None:
                continue

            # For 'downsample_gallery' degrade the gallery image to low_res then upscale
            if method == 'downsample_gallery':
                proc = downsample_then_upscale(crop, low_res=low_res, upscale_to=embed_size)
            else:
                # Default: resize gallery crop to embedding size
                proc = resize_keep_aspect(crop, embed_size)

            emb = get_embedding_for_face(proc)
            if emb is not None:
                chosen_proc = proc
                chosen_path = path
                break  # gallery embedding obtained for this identity

        # Skip identity if no gallery embedding found
        if emb is None:
            continue
        gallery_embs[name] = emb
        gallery_samples[name] = (chosen_path, chosen_proc)

    # Process probes and compute probe embeddings based on selected method
    probe_list = []
    example_count = 0

    # Counters for strict naive diagnostics
    strict_attempted = 0
    strict_embedded = 0
    strict_skipped = 0

    for name, path in tqdm(probes, desc="Probes"):
        # Skip probes when gallery embedding for that identity doesn't exist
        if name not in gallery_embs:
            continue

        img = cv2.imread(path)
        if img is None:
            pil_rgb = safe_load_image(path)
            if pil_rgb is None:
                continue
            img = cv2.cvtColor(pil_rgb, cv2.COLOR_RGB2BGR)

        crop = detect_and_crop_face(img, img_path=path)
        if crop is None:
            continue

        proc = None

        # ----------------------------
        # Method-specific processing
        # ----------------------------
        if method == 'naive':
            if strict_naive:
                # STRICT NAIVE:
                # 1) Downscale to low_res to simulate CCTV crop.
                low = cv2.resize(crop, low_res, interpolation=cv2.INTER_AREA)
                # 2) Upscale with NEAREST to keep blocky low-res appearance.
                proc = cv2.resize(low, embed_size, interpolation=cv2.INTER_NEAREST)

                strict_attempted += 1
                emb = get_embedding_for_face(proc)
                if emb is None:
                    # Skip probe if embedding fails under strict naive (no fallback).
                    strict_skipped += 1
                    continue
                else:
                    strict_embedded += 1
            else:
                # Non-strict naive (legacy): resize probe to embed size (may inflate performance)
                proc = resize_keep_aspect(crop, embed_size)
                emb = get_embedding_for_face(proc)
                if emb is None:
                    # Legacy fallback to try producing an embedding (keeps results permissive)
                    emb = get_embedding_for_face(resize_keep_aspect(crop, embed_size))
                    if emb is None:
                        continue

        elif method == 'downsample_gallery':
            # Keep probe at high-res and gallery was downsampled earlier.
            proc = resize_keep_aspect(crop, embed_size)
            emb = get_embedding_for_face(proc)
            if emb is None:
                emb = get_embedding_for_face(resize_keep_aspect(crop, embed_size))
                if emb is None:
                    continue

        elif method == 'sr_probe':
            # Simulate probe low-res then apply DNN SR if available, else bicubic
            low = cv2.resize(crop, low_res, interpolation=cv2.INTER_AREA)
            if sr is not None:
                up = sr_upscale_with_model(sr, low)
                proc = resize_keep_aspect(up, embed_size)
            else:
                proc = cv2.resize(low, embed_size, interpolation=cv2.INTER_CUBIC)
            emb = get_embedding_for_face(proc)
            if emb is None:
                # Fallback to resized crop if SR/bicubic embedding fails
                emb = get_embedding_for_face(resize_keep_aspect(crop, embed_size))
                if emb is None:
                    continue

        elif method == 'bicubic_probe':
            # Downscale probe then bicubic upsample to embedding size
            low = cv2.resize(crop, low_res, interpolation=cv2.INTER_AREA)
            proc = cv2.resize(low, embed_size, interpolation=cv2.INTER_CUBIC)
            emb = get_embedding_for_face(proc)
            if emb is None:
                emb = get_embedding_for_face(resize_keep_aspect(crop, embed_size))
                if emb is None:
                    continue

        else:
            # Unknown method -> skip
            continue

        # Defensive check for embedding existence (non-strict paths)
        if method != 'naive' or not strict_naive:
            if 'emb' not in locals() or emb is None:
                continue

        # Append valid probe tuple: (true_name, embedding, processed_image, image_path)
        probe_list.append((name, emb, proc, path))

        # Save a side-by-side example (gallery vs processed probe) if requested
        if save_examples and example_count < max_examples:
            gpath, gimg = gallery_samples.get(name, (None, None))
            if gimg is not None:
                h1, w1 = gimg.shape[:2]
                h2, w2 = proc.shape[:2]
                h = max(h1, h2)
                canvas = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
                canvas[:h1, :w1] = cv2.resize(gimg, (w1, h1))
                canvas[:h2, w1:w1 + w2] = cv2.resize(proc, (w2, h2))
                fn = os.path.join(examples_dir, f"{name}_example_{example_count}.png")
                cv2.imwrite(fn, canvas)
                example_count += 1

    # Print summary counts for strict naive mode (demonstrates how many probes skipped)
    if strict_naive and method == 'naive':
        print(f"[strict_naive] probes attempted: {strict_attempted}, embedded: {strict_embedded}, skipped: {strict_skipped}")

    return gallery_embs, probe_list

def evaluate_rank1(gallery_embs, probe_list):
    """Compute Rank-1 accuracy and collect genuine/impostor similarity arrays for ROC."""
    names = list(gallery_embs.keys())
    if len(names) == 0:
        return 0.0, np.array([]), np.array([])
    X = np.array([gallery_embs[n] for n in names])
    correct = 0
    total = 0
    sim_genuine = []
    sim_impostor = []
    for pname, p_emb, _, _ in probe_list:
        total += 1
        sims = cosine_similarity(p_emb.reshape(1, -1), X).flatten()
        idx = sims.argmax()
        pred = names[idx]
        if pred == pname:
            correct += 1
        for i, gname in enumerate(names):
            if gname == pname:
                sim_genuine.append(sims[i])
            else:
                sim_impostor.append(sims[i])
    rank1 = correct / total if total > 0 else 0.0
    return rank1, np.array(sim_genuine), np.array(sim_impostor)

def append_result_csv(out_csv, row):
    """Append one result row to CSV, creating header if file doesn't exist."""
    header = ['timestamp','method','seed','max_id','low_res','gallery_size','probes','rank1','auc']
    write_header = not os.path.exists(out_csv)
    with open(out_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

def main():
    """Parse CLI args, prepare SR model (optional), run processing, compute metrics, and save outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='lfw_dataset_fixed', help='path to dataset folder (each identity subfolder)')
    parser.add_argument('--max_id', type=int, default=50, help='max identities to test')
    parser.add_argument('--low_res', type=int, default=24, help='low resolution side for probe (square)')
    parser.add_argument('--method', choices=['naive','downsample_gallery','bicubic_probe','sr_probe'], default='naive')
    parser.add_argument('--sr_model', type=str, default=None, help='path to opencv dnn sr model (.pb)')
    parser.add_argument('--sr_name', type=str, default='edsr', help='name for sr model (edsr, fsrcnn, lapsrn...)')
    parser.add_argument('--sr_scale', type=int, default=4)
    parser.add_argument('--save_examples', action='store_true', help='save a few gallery vs probe example images')
    parser.add_argument('--dump_ids', type=str, default=None, help='path to write chosen identity names (one per line)')
    parser.add_argument('--seed', type=int, default=42, help='random seed for deterministic identity selection')
    parser.add_argument('--strict_naive', action='store_true', help='For naive method: keep probes at low_res (no upscaling) to simulate true mismatch')
    parser.add_argument('--out_csv', type=str, default=None, help='append result line (CSV) for comparison across runs')
    args = parser.parse_args()

    # Build deterministic gallery and probe lists
    gallery, probes = build_gallery_and_probes(args.dataset, max_identities=args.max_id, seed=args.seed, dump_ids=args.dump_ids)

    # Load SR model if requested
    sr = None
    if args.method == 'sr_probe' and args.sr_model:
        sr = make_dnn_sr(args.sr_model, sr_name=args.sr_name, scale=args.sr_scale)

    # Process images and obtain embeddings
    gallery_embs, probe_list = process_and_embed(gallery, probes, sr=sr, method=args.method,
                                                 low_res=(args.low_res, args.low_res), embed_size=(160,160),
                                                 strict_naive=args.strict_naive, save_examples=args.save_examples)

    # Evaluate Rank-1 and ROC/AUC
    rank1, genuine, impostor = evaluate_rank1(gallery_embs, probe_list)
    print(f"Method={args.method} | Rank-1: {rank1*100:.2f}% | Identities tested: {len(gallery_embs)} | Probes used: {len(probe_list)}")

    roc_auc = None
    if genuine.size == 0 and impostor.size == 0:
        print("No sim scores to compute ROC (not enough data).")
    else:
        y = np.concatenate([np.ones_like(genuine), np.zeros_like(impostor)])
        scores = np.concatenate([genuine, impostor])
        fpr, tpr, thr = roc_curve(y, scores)
        roc_auc = auc(fpr, tpr)
        print("AUC:", roc_auc)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC={roc_auc:.3f}')
        plt.plot([0,1], [0,1], linestyle='--', label='chance')
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.title(f'ROC - {args.method}')
        outname = f'roc_{args.method}_seed{args.seed}_maxid{args.max_id}.png'
        plt.savefig(outname)
        print("Saved", outname)

    # Optionally append result row to CSV for comparison/aggregation
    if args.out_csv:
        ts = datetime.datetime.now().isoformat()
        row = [ts, args.method, args.seed, args.max_id, args.low_res, len(gallery_embs), len(probe_list), f"{rank1:.6f}", f"{roc_auc:.6f}" if roc_auc is not None else ""]
        try:
            append_result_csv(args.out_csv, row)
            print("Appended results to", args.out_csv)
        except Exception as e:
            print("Failed to append CSV:", e)

if __name__ == '__main__':
    main()
