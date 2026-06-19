import numpy as np
from insightface.app import FaceAnalysis


def build_face_analyzer(model_root, ctx_id=0, det_thresh=0.3, det_size=(640, 640)):
    app = FaceAnalysis(name="antelope", root=str(model_root))
    try:
        app.prepare(ctx_id=ctx_id, det_thresh=det_thresh, det_size=det_size)
    except Exception:
        app.prepare(ctx_id=-1, det_thresh=det_thresh, det_size=det_size)
    return app


def largest_face(faces):
    if faces is None or len(faces) == 0:
        return None
    return max(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))


def get_face_embedding(app, image_bgr):
    face = largest_face(app.get(image_bgr))
    if face is None:
        return None

    embedding = getattr(face, "normed_embedding", None)
    if embedding is None:
        embedding = getattr(face, "embedding", None)
    if embedding is None:
        return None

    embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
    embedding = embedding / (np.linalg.norm(embedding) + 1e-12)
    return embedding


def cosine_similarity(a, b):
    if a is None or b is None:
        return None
    return float(np.dot(a, b))


def detect_largest_face(app, image_bgr):
    return largest_face(app.get(image_bgr))
