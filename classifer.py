"""
AI Sheep Analyzer — Classify Backend (Hugging Face Spaces)
Handles: /health /classify
Pipeline: YOLO → Custom EfficientNetV2-ONNX → CLIP Ensemble → Pixel Fallback
"""
import argparse
import io
import logging
import math
import sys
import time
import traceback
import warnings
from contextlib import asynccontextmanager, contextmanager

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")

try:
    import multipart  # noqa: F401
except ImportError:
    sys.exit("[ERROR] pip install python-multipart")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sheep.classify")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BREEDS           = ["Lohi", "Kajli", "Lari"]
MAX_UPLOAD       = 15 * 1024 * 1024
VALID_BREEDS     = {"Lohi", "Kajli", "Lari"}
VALID_AGES       = {"lamb", "young", "adult", "old"}
VALID_HEALTH     = {"healthy", "weak", "diseased"}
CUSTOM_THRESHOLD = 0.75

BREED_META = {
    "Lohi":  ("Sindh/Punjab",    "Wool, Meat"),
    "Kajli": ("Sindh/Punjab",    "Meat, Breeding"),
    "Lari":  ("Sindh (coastal)", "Meat"),
}

BREED_PROMPTS = {
    "Lohi": [
        "a Lohi sheep with light tan brown woolly face from Punjab Pakistan",
        "Pakistani Lohi sheep breed with tan pinkish face and thick white wool body",
        "a wool and meat sheep from Punjab Pakistan with light brownish tan facial coloring",
        "Lohi breed sheep with characteristic tan face and fine wool fleece from Pakistani Punjab",
    ],
    "Kajli": [
        "a Kajli sheep with bright white face and Roman nose from Sindh Pakistan",
        "Pakistani Kajli sheep with pure white face long Roman nose and tall long legged body",
        "a white faced sheep from Sindh Punjab Pakistan with distinctive Roman curved nose profile",
        "Kajli breed sheep with white face dark eye rings and tall elegant build from Pakistan",
    ],
    "Lari": [
        "a Lari sheep with faded brown coastal face from Sindh Pakistan",
        "Pakistani Lari sheep also called Jhari with desaturated light brown face from coastal Sindh",
        "a coastal Sindh sheep breed with faded brownish greyish face and heavy body frame",
        "Lari Jhari breed sheep with characteristic faded coastal coloring from southern Sindh Pakistan",
    ],
}

AGE_PROMPTS = {
    "lamb":  ["a newborn lamb sheep very small and young under 6 months old",
              "a baby lamb sheep just born still small and fragile",
              "a very young sheep lamb under six months of age",
              "a juvenile baby sheep lamb with small body"],
    "young": ["a young sheep between 6 and 18 months old juvenile",
              "a juvenile sheep adolescent between six and eighteen months",
              "a teenage sheep not yet fully grown between 6 to 18 months",
              "a young sheep not yet adult aged 6 to 18 months"],
    "adult": ["a fully grown adult sheep in prime condition 1.5 to 4 years old",
              "a mature adult sheep in peak physical condition",
              "a prime age sheep fully developed between 2 and 4 years old",
              "an adult sheep in its prime years fully grown"],
    "old":   ["an old aged sheep with worn features over 4 years old",
              "an elderly sheep past prime with aged features",
              "an old sheep more than 4 years with visible aging signs",
              "a geriatric sheep very old past productive age"],
}

HEALTH_PROMPTS = {
    "healthy":  ["a healthy sheep with vibrant coat and alert posture in good condition",
                 "a fit healthy sheep with shiny coat standing alert",
                 "a sheep in excellent health with bright eyes and clean coat",
                 "a vigorous healthy sheep with good body condition"],
    "weak":     ["a weak sheep with dull coat and low energy appearing lethargic",
                 "a sheep appearing weak and lethargic with poor coat condition",
                 "a visibly tired weak sheep with dull eyes and thin body",
                 "a sheep in poor condition appearing weak and undernourished"],
    "diseased": ["a sick diseased sheep with visible health problems and suffering",
                 "a sheep showing clear signs of illness and disease",
                 "a visibly ill sheep with sickness symptoms and poor condition",
                 "a diseased sheep clearly unwell with visible health issues"],
}

CLIP_REASON = {
    "Lohi":  "CLIP detected tan/brown woolly face — Lohi (Punjab/Sindh wool breed)",
    "Kajli": "CLIP detected bright white face with Roman nose — Kajli",
    "Lari":  "CLIP detected faded coastal colouring with heavy body — Lari (coastal Sindh)",
}

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

YOLO_MODEL        = None
CLIP_MODEL        = None
CLIP_MODEL_2      = None
CLIP_PREPROCESS   = None
CLIP_PREPROCESS_2 = None
CUSTOM_MODEL      = None
CUSTOM_PREPROCESS = None
DEVICE            = "cpu"
CLIP_DEVICE       = "cpu"

try:
    import torch
    if torch.cuda.is_available():
        gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        DEVICE = CLIP_DEVICE = "cuda"
        log.info(f"GPU: {gpu_gb:.1f}GB — {'full' if gpu_gb >= 8 else 'partial'} GPU mode")
    else:
        log.info("No GPU — CPU mode")
except ImportError:
    log.warning("PyTorch not installed — pixel fallback only")

try:
    from ultralytics import YOLO
    YOLO_MODEL = YOLO("yolov8x.pt")
    YOLO_MODEL.to(DEVICE)
    log.info(f"✓ YOLOv8x on {DEVICE}")
except Exception as e:
    log.warning(f"✗ YOLOv8x: {e}")

try:
    import clip
    CLIP_MODEL, CLIP_PREPROCESS = clip.load("ViT-L/14@336px", device=CLIP_DEVICE)
    log.info(f"✓ CLIP ViT-L/14@336px on {CLIP_DEVICE}")
    try:
        CLIP_MODEL_2, CLIP_PREPROCESS_2 = clip.load("ViT-L/14", device=CLIP_DEVICE)
        log.info("✓ CLIP ViT-L/14 secondary")
    except Exception as e:
        log.warning(f"✗ CLIP secondary: {e}")
except Exception as e:
    log.warning(f"✗ CLIP: {e}")

try:
    import onnxruntime as ort
    _providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                  if "CUDAExecutionProvider" in ort.get_available_providers()
                  else ["CPUExecutionProvider"])
    CUSTOM_MODEL = ort.InferenceSession("sheep_breed_classifier.onnx", providers=_providers)
    log.info(f"✓ Custom EfficientNetV2-ONNX (providers={_providers})")
except FileNotFoundError:
    log.warning("✗ sheep_breed_classifier.onnx not found — run train_custom.py first")
except Exception as e:
    log.warning(f"✗ Custom model: {e}")

try:
    import albumentations as A
    CUSTOM_PREPROCESS = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    log.info("✓ Albumentations preprocessing ready")
except ImportError:
    log.warning("✗ Albumentations not installed")

# ---------------------------------------------------------------------------
# Pixel fallback helpers
# ---------------------------------------------------------------------------

def _rgb(img: Image.Image):
    a = np.array(img.resize((64, 64)).convert("RGB"), dtype=np.float32)
    return a[:, :, 0].mean(), a[:, :, 1].mean(), a[:, :, 2].mean()

def _region(img: Image.Image, x0, y0, x1, y1):
    w, h = img.size
    a = np.array(
        img.crop((int(w * x0), int(h * y0), int(w * x1), int(h * y1))).convert("RGB"),
        dtype=np.float32,
    )
    return a[:, :, 0].mean(), a[:, :, 1].mean(), a[:, :, 2].mean()

def _br(r, g, b):           return (r + g + b) / 3
def _wht(r, g, b, t=185):   return r > t and g > t and b > t
def _tan(r, g, b):          return r > 140 and r > g * 1.08 and g > 100 and b < 155 and r < 210
def _fade(r, g, b):         return 90 < _br(r, g, b) < 175 and (max(r, g, b) - min(r, g, b)) < 55


def _pixel_fallback(img: Image.Image) -> dict:
    fr, fg, fb = _region(img, .25, .0, .75, .30)
    w, h = img.size

    eye_c = (
        np.array(img.crop((int(w * .40), int(h * .05), int(w * .60), int(h * .20))).convert("RGB"), dtype=np.float32).mean()
        - (np.array(img.crop((int(w * .28), int(h * .12), int(w * .40), int(h * .24))).convert("RGB"), dtype=np.float32).mean()
         + np.array(img.crop((int(w * .60), int(h * .12), int(w * .72), int(h * .24))).convert("RGB"), dtype=np.float32).mean()) / 2
    )
    wool      = float(np.array(img.crop((int(w * .15), int(h * .60), int(w * .85), int(h * .85))).convert("L"), dtype=np.float32).std())
    body_fill = np.array(img.crop((int(w * .1), int(h * .55), int(w * .9), int(h * .90))).convert("RGB"), dtype=np.float32).mean()

    if   _wht(fr, fg, fb, 175) and eye_c > 18: breed, conf, reason = "Kajli", 0.87, "White face with dark eye-ring contrast"
    elif _wht(fr, fg, fb, 178):                breed, conf, reason = "Kajli", 0.76, "Pure white face — Kajli type"
    elif _tan(fr, fg, fb) and wool > 38:        breed, conf, reason = "Lohi",  0.85, "Tan face + wool texture — Lohi"
    elif _tan(fr, fg, fb):                      breed, conf, reason = "Lohi",  0.75, "Light tan face — Lohi phenotype"
    elif _fade(fr, fg, fb) and body_fill > 145: breed, conf, reason = "Lari",  0.78, "Faded coastal face + heavy body — Lari"
    else:                                       breed, conf, reason = "Lari",  0.52, "Faded light-brown face — Lari coastal"

    ob            = _br(*_rgb(img))
    age, ac       = (("lamb", 0.55) if ob > 190 else ("young", 0.55) if ob > 155 else ("adult", 0.60) if ob > 110 else ("old", 0.52))
    r, g, b       = _rgb(img)
    sat           = float(max(r, g, b) - min(r, g, b))
    health, hc    = ("healthy", 0.65) if sat > 55 else ("weak", 0.58) if sat > 25 else ("diseased", 0.52)

    rem = [x for x in BREEDS if x != breed]
    sp  = (1 - conf) / max(len(rem), 1)
    return {
        "is_sheep":         True,
        "breed":            breed,
        "breed_confidence": round(conf, 3),
        "top_breeds":       [{"breed": breed, "probability": round(conf, 3)}]
                          + [{"breed": x, "probability": round(sp, 3)} for x in rem[:2]],
        "age":              age,
        "age_confidence":   round(ac, 3),
        "health":           health,
        "health_confidence":round(hc, 3),
        "yolo_crop":        False,
        "bcs":              3 if health == "healthy" else (2 if health == "weak" else 1),
        "wool":             "Fine" if breed == "Lohi" else "Medium",
        "use":              BREED_META[breed][1],
        "pipeline":         "Pixel fallback",
        "reason":           reason,
        "_price_breed":     breed,
    }

# ---------------------------------------------------------------------------
# CLIP helpers
# ---------------------------------------------------------------------------

def _clip_score(model, preprocess, img_pil: Image.Image, prompt_dict: dict, device: str) -> dict:
    import torch
    import clip as clip_lib
    img_t = preprocess(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        img_f = model.encode_image(img_t)
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        return {
            label: sum(
                (img_f @ (
                    model.encode_text(clip_lib.tokenize([p]).to(device))
                    / model.encode_text(clip_lib.tokenize([p]).to(device)).norm(dim=-1, keepdim=True)
                ).T).item()
                for p in prompts
            ) / len(prompts)
            for label, prompts in prompt_dict.items()
        }


def _softmax_probs(scores: dict, temp: float = 5.0) -> dict:
    exp_v = {k: math.exp(v * temp) for k, v in scores.items()}
    total = sum(exp_v.values())
    return {k: v / total for k, v in exp_v.items()}


def _confidence_from_scores(scores: dict) -> float:
    vals = list(scores.values())
    mean = sum(vals) / len(vals)
    return round(min(0.92, max(0.55, max(vals) / (mean + 1e-6) * 0.5)), 3)

# ---------------------------------------------------------------------------
# Custom ONNX model inference
# ---------------------------------------------------------------------------

def _softmax_np(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _custom_classify(img_pil: Image.Image) -> dict | None:
    """Returns result dict if confidence >= CUSTOM_THRESHOLD, else None."""
    if CUSTOM_MODEL is None or CUSTOM_PREPROCESS is None:
        return None
    try:
        img_np    = np.array(img_pil.convert("RGB").resize((224, 224)), dtype=np.float32)
        arr       = CUSTOM_PREPROCESS(image=img_np)["image"]   # HWC normalized
        arr       = np.expand_dims(np.transpose(arr, (2, 0, 1)), 0).astype(np.float32)  # NCHW
        raw       = CUSTOM_MODEL.run(None, {CUSTOM_MODEL.get_inputs()[0].name: arr})[0][0]
        probs     = _softmax_np(raw)
        top_idx   = int(np.argmax(probs))
        top_conf  = float(probs[top_idx])

        if top_conf < CUSTOM_THRESHOLD:
            return None

        return {
            "breed":            BREEDS[top_idx],
            "breed_confidence": round(top_conf, 3),
            "top_breeds":       [{"breed": BREEDS[i], "probability": round(float(p), 3)}
                                 for i, p in sorted(enumerate(probs), key=lambda x: x[1], reverse=True)],
            "pipeline":         f"Custom EfficientNetV2-ONNX [conf:{round(top_conf, 3)}]",
            "reason":           f"Fine-tuned model — {BREEDS[top_idx]} with {round(top_conf * 100, 1)}% confidence",
        }
    except Exception as e:
        log.warning(f"Custom model inference failed: {e}")
        return None

# ---------------------------------------------------------------------------
# Output validator
# ---------------------------------------------------------------------------

def _validate(r: dict) -> dict:
    if r.get("breed")  not in VALID_BREEDS: r["breed"]  = "Lari"
    if r.get("age")    not in VALID_AGES:   r["age"]    = "adult"
    if r.get("health") not in VALID_HEALTH: r["health"] = "healthy"
    for k in ("breed_confidence", "age_confidence", "health_confidence"):
        r[k] = round(min(1.0, max(0.0, float(r.get(k, 0.5)))), 3)
    r["bcs"] = max(1, min(5, int(r.get("bcs", 3))))
    if not r.get("top_breeds"):
        b, c = r["breed"], r["breed_confidence"]
        rem  = [x for x in BREEDS if x != b]
        sp   = (1 - c) / max(len(rem), 1)
        r["top_breeds"] = [{"breed": b, "probability": round(c, 3)}] \
                        + [{"breed": x, "probability": round(sp, 3)} for x in rem[:2]]
    return r

# ---------------------------------------------------------------------------
# Image quality check
# ---------------------------------------------------------------------------

def _image_quality(img: Image.Image) -> dict:
    try:
        gray  = np.array(img.convert("L"), dtype=np.float32)
        lum   = float(gray.mean())
        sharp = (float(cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var())
                 if CV2_AVAILABLE
                 else float(np.var(gray[1:] - gray[:-1]) + np.var(gray[:, 1:] - gray[:, :-1])))
        return {
            "is_blurry":       sharp < 80,
            "is_overexposed":  lum > 220,
            "is_underexposed": lum < 35,
            "sharpness":       round(sharp, 1),
            "luminance":       round(lum, 1),
        }
    except Exception:
        return {"is_blurry": False, "is_overexposed": False,
                "is_underexposed": False, "sharpness": 0, "luminance": 128}

# ---------------------------------------------------------------------------
# Main classification pipeline
# ---------------------------------------------------------------------------

def _classify(img: Image.Image) -> dict:
    if CLIP_MODEL is None:
        return _pixel_fallback(img)

    import torch

    # Layer 1 — YOLO crop
    sheep_img, yolo_crop = img, False
    if YOLO_MODEL is not None:
        try:
            results = YOLO_MODEL(img, verbose=False)
            best = max(
                ((box, float(box.conf[0])) for r in results for box in r.boxes
                 if YOLO_MODEL.names.get(int(box.cls[0].item())) == "sheep"),
                key=lambda x: x[1], default=(None, 0),
            )
            if best[0] is not None:
                b = best[0].xyxy[0].tolist()
                w, h = img.size
                pad  = 0.10
                sheep_img = img.crop((
                    max(0, int(b[0] - w * pad)), max(0, int(b[1] - h * pad)),
                    min(w, int(b[2] + w * pad)), min(h, int(b[3] + h * pad)),
                ))
                yolo_crop = True
            else:
                return _validate({
                    "is_sheep": False, "error": "no_sheep_detected",
                    "breed": "Unknown", "breed_confidence": 0.0, "top_breeds": [],
                    "age": "unknown", "age_confidence": 0.0,
                    "health": "unknown", "health_confidence": 0.0,
                    "yolo_crop": False, "bcs": 0, "wool": "—", "use": "—",
                    "pipeline": "YOLOv8x — no sheep detected",
                    "reason": "No sheep found", "_price_breed": "Lohi",
                })
        except Exception as e:
            log.warning(f"YOLO failed: {e}")

    # Layer 2 — Custom ONNX model
    custom = _custom_classify(sheep_img)
    if custom is not None:
        body_img      = sheep_img.resize((336, 336), Image.LANCZOS)
        age_scores    = _clip_score(CLIP_MODEL, CLIP_PREPROCESS, body_img, AGE_PROMPTS, CLIP_DEVICE)
        health_scores = _clip_score(CLIP_MODEL, CLIP_PREPROCESS, body_img, HEALTH_PROMPTS, CLIP_DEVICE)
        age           = max(age_scores, key=age_scores.get)
        health        = max(health_scores, key=health_scores.get)
        breed         = custom["breed"]
        return _validate({
            "is_sheep":          True,
            "breed":             breed,
            "breed_confidence":  custom["breed_confidence"],
            "top_breeds":        custom["top_breeds"],
            "age":               age,
            "age_confidence":    _confidence_from_scores(age_scores),
            "health":            health,
            "health_confidence": _confidence_from_scores(health_scores),
            "yolo_crop":         yolo_crop,
            "bcs":               3 if health == "healthy" else (2 if health == "weak" else 1),
            "wool":              "Fine" if breed == "Lohi" else "Medium",
            "use":               BREED_META.get(breed, ("Sindh", "Meat"))[1],
            "pipeline":          custom["pipeline"],
            "reason":            custom["reason"],
            "_price_breed":      breed,
        })

    # Layer 3 — CLIP ensemble
    sw, sh   = sheep_img.size
    face_img = sheep_img.crop((int(sw * .05), 0, int(sw * .95), int(sh * .38))).resize((336, 336), Image.LANCZOS)
    body_img = sheep_img.resize((336, 336), Image.LANCZOS)

    s1 = _clip_score(CLIP_MODEL, CLIP_PREPROCESS, face_img, BREED_PROMPTS, CLIP_DEVICE)
    s2 = (_clip_score(CLIP_MODEL_2, CLIP_PREPROCESS_2,
                      sheep_img.crop((int(sw * .05), 0, int(sw * .95), int(sh * .38))),
                      BREED_PROMPTS, CLIP_DEVICE)
          if CLIP_MODEL_2 else {})
    px = _pixel_fallback(img)
    ps = {b: 0.05 for b in BREEDS}
    ps[px["breed"]] = px["breed_confidence"] * 0.35

    def _norm(scores):
        mn, mx = min(scores.values()), max(scores.values())
        rng = max(mx - mn, 1e-6)
        return {k: (v - mn) / rng for k, v in scores.items()}

    ensemble: dict[str, float] = {b: 0.0 for b in BREEDS}
    if s1:
        for b in BREEDS: ensemble[b] += _norm(s1)[b] * 0.55
    if s2:
        for b in BREEDS: ensemble[b] += _norm(s2)[b] * 0.30
    for b in BREEDS:
        ensemble[b] += ps.get(b, 0) * 0.15

    probs   = _softmax_probs(ensemble)
    breed   = max(probs, key=probs.get)
    breed_c = round(probs[breed], 3)

    if breed_c < 0.35:
        breed, breed_c = px["breed"], px["breed_confidence"]
        pipeline = f"Pixel fallback [low conf:{breed_c}]"
    else:
        pipeline = f"CLIP-L14@336 ensemble [{CLIP_DEVICE}] conf:{breed_c}"

    age_scores    = _clip_score(CLIP_MODEL, CLIP_PREPROCESS, body_img, AGE_PROMPTS, CLIP_DEVICE)
    health_scores = _clip_score(CLIP_MODEL, CLIP_PREPROCESS, body_img, HEALTH_PROMPTS, CLIP_DEVICE)
    age    = max(age_scores, key=age_scores.get)
    health = max(health_scores, key=health_scores.get)

    return _validate({
        "is_sheep":          True,
        "breed":             breed,
        "breed_confidence":  breed_c,
        "top_breeds":        [{"breed": b, "probability": round(p, 3)}
                              for b, p in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]],
        "age":               age,
        "age_confidence":    _confidence_from_scores(age_scores),
        "health":            health,
        "health_confidence": _confidence_from_scores(health_scores),
        "yolo_crop":         yolo_crop,
        "bcs":               3 if health == "healthy" else (2 if health == "weak" else 1),
        "wool":              "Fine" if breed == "Lohi" else "Medium",
        "use":               BREED_META.get(breed, ("Sindh", "Meat"))[1],
        "pipeline":          pipeline,
        "reason":            CLIP_REASON.get(breed, ""),
        "_price_breed":      breed,
    })

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=" * 60)
    log.info("AI Sheep Analyzer — Classify Backend (Hugging Face Spaces)")
    log.info(f"  YOLO:         {'✓ YOLOv8x' if YOLO_MODEL   else '✗ pixel fallback'}")
    log.info(f"  Custom ONNX:  {'✓ EfficientNetV2' if CUSTOM_MODEL else '✗ not loaded'}")
    log.info(f"  CLIP-1:       {'✓ ViT-L/14@336px' if CLIP_MODEL  else '✗ pixel fallback'}")
    log.info(f"  CLIP-2:       {'✓ ViT-L/14' if CLIP_MODEL_2 else '✗ not loaded'}")
    log.info(f"  Device:       {DEVICE}")
    log.info("=" * 60)
    if CLIP_MODEL:
        try:
            _classify(Image.new("RGB", (336, 336), (128, 128, 128)))
            log.info("✓ Warmup done")
        except Exception as e:
            log.warning(f"Warmup failed: {e}")
    yield


app = FastAPI(title="AI Sheep Analyzer — Classify Backend", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@contextmanager
def _nullctx():
    yield


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "backend":      "classify",
        "yolo":         YOLO_MODEL is not None,
        "custom_model": CUSTOM_MODEL is not None,
        "clip":         CLIP_MODEL is not None,
        "clip2":        CLIP_MODEL_2 is not None,
        "device":       DEVICE,
        "pipeline":     "YOLO → Custom → CLIP → Pixel",
    }


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    ct = (file.content_type or "").lower().split(";")[0].strip()
    if ct and not ct.startswith("image/"):
        raise HTTPException(415, f"Expected image, got: {ct}")

    raw = await file.read()
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, "File too large — max 15MB")

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    try:
        t0     = time.perf_counter()
        result = _classify(img)
        result.pop("_price_breed", None)
        return JSONResponse({
            **result,
            "quality":    _image_quality(img),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000),
        })
    except Exception as e:
        log.error(traceback.format_exc())
        raise HTTPException(500, f"Classification failed: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port, reload=False)
