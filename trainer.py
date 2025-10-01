import os
import json
import time
import glob
from typing import Dict, List, Tuple

import cv2
import numpy as np

from face_engine import FaceEngine

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')
CENTROIDS_PATH = os.path.join(MODELS_DIR, 'centroids.npz')
META_PATH = os.path.join(MODELS_DIR, 'meta.json')

THRESHOLD_SIM = 0.38  # similarity threshold; tune 0.35–0.45 depending on your data


def ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)


def _list_images(person_dir: str) -> List[str]:
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(person_dir, e)))
    return sorted(files)


def build_centroids(det_size=(640, 640), ctx_id=0) -> Tuple[Dict[str, np.ndarray], Dict]:
    """Scan dataset/ and build a centroid (mean embedding) per person folder."""
    ensure_dirs()
    fe = FaceEngine(det_size=det_size, ctx_id=ctx_id)

    label_to_embs: Dict[str, List[np.ndarray]] = {}

    people = [d for d in sorted(os.listdir(DATASET_DIR)) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    for person in people:
        pdir = os.path.join(DATASET_DIR, person)
        imgs = _list_images(pdir)
        if not imgs:
            continue
        for path in imgs:
            img = cv2.imread(path)
            if img is None:
                continue
            face = fe.get_best_face(img)
            if face is None or face.normed_embedding is None:
                continue
            emb = face.normed_embedding.astype(np.float32)
            label_to_embs.setdefault(person, []).append(emb)

    centroids = {}
    for label, embs in label_to_embs.items():
        if len(embs) == 0:
            continue
        arr = np.stack(embs, axis=0)
        # embeddings from insightface are already normalized; still normalize centroid for safety
        centroid = arr.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
        centroids[label] = centroid

    meta = {
        "created_at": int(time.time()),
        "num_classes": len(centroids),
        "threshold_sim": THRESHOLD_SIM,
    }

    return centroids, meta


def save_centroids(centroids: Dict[str, np.ndarray], meta: Dict):
    ensure_dirs()
    # Save as npz
    np.savez_compressed(CENTROIDS_PATH, **centroids)
    with open(META_PATH, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)


def load_centroids() -> Tuple[Dict[str, np.ndarray], Dict]:
    if not os.path.exists(CENTROIDS_PATH) or not os.path.exists(META_PATH):
        return {}, {"threshold_sim": THRESHOLD_SIM}
    data = np.load(CENTROIDS_PATH)
    centroids = {k: data[k] for k in data.files}
    with open(META_PATH, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    if 'threshold_sim' not in meta:
        meta['threshold_sim'] = THRESHOLD_SIM
    return centroids, meta


def predict_name(embedding: np.ndarray, centroids: Dict[str, np.ndarray], threshold_sim: float) -> Tuple[str, float]:
    if not centroids:
        return "(no model)", 0.0
    best_label = "Unknown"
    best_sim = -1.0
    for label, c in centroids.items():
        sim = FaceEngine.cosine_similarity(embedding, c)
        if sim > best_sim:
            best_sim = sim
            best_label = label
    if best_sim < threshold_sim:
        return "Unknown", best_sim
    return best_label, best_sim


def rebuild_model(det_size=(640, 640), ctx_id=0) -> Dict:
    centroids, meta = build_centroids(det_size=det_size, ctx_id=ctx_id)
    save_centroids(centroids, meta)
    return {"classes": list(centroids.keys()), "meta": meta}

def delete_person(name: str) -> bool:
    """Delete a person's folder from dataset/. Returns True if deleted, False if not found."""
    pdir = os.path.join(DATASET_DIR, name)
    if not os.path.exists(pdir) or not os.path.isdir(pdir):
        return False
    # Remove all files in the directory
    files = _list_images(pdir)
    for f in files:
        try:
            os.remove(f)
        except Exception:
            pass
    # Remove the directory itself
    try:
        os.rmdir(pdir)
    except Exception:
        pass
    return True