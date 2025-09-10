import time
import cv2
import numpy as np
from typing import List, Tuple, Optional



try:
    # Core FaceAnalysis from InsightFace (ArcFace embeddings via models in the 'buffalo_l' pack)
    from insightface.app import FaceAnalysis
except Exception as e:
    raise RuntimeError(
        "InsightFace import failed. Install with `pip install insightface` and ensure requirements are met."
    ) from e




class FaceEngine:
    """Wrapper around InsightFace FaceAnalysis for detection + ArcFace embeddings."""


    def __init__(self, det_size: Tuple[int, int] = (640, 640), ctx_id: int = 0):
        # ctx_id: 0 = GPU if available, else CPU fallback; set to -1 to force CPU
        self.app = FaceAnalysis(name='buffalo_l') # includes retinaface det + arcface
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)


    def detect(self, bgr_image: np.ndarray):
        """Return a list of faces with bbox, landmarks, detection score, and embeddings."""
        return self.app.get(bgr_image)


    def get_best_face(self, bgr_image: np.ndarray):
        faces = self.detect(bgr_image)
        if not faces:
            return None
        # choose the highest detection score
        faces.sort(key=lambda f: float(f.det_score), reverse=True)
        return faces[0]


    @staticmethod
    def draw_bbox_with_label(img: np.ndarray, bbox: np.ndarray, label: str):
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        y_text = max(0, y1 - 10)
        cv2.putText(img, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """ArcFace embeddings are typically L2-normalized; still guard against div-by-zero."""
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return -1.0
        return float(np.dot(a, b) / denom)  
    

    