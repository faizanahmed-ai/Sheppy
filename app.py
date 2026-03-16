import argparse, datetime, io, logging, sys, time, traceback, warnings
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

warnings.filterwarnings("ignore")

# ── Optional deps ─────────────────────────────────────────────────
try:
    import multipart
except ImportError:
    sys.exit("\n[ERROR] pip install python-multipart\n")

try:
    import cv2; CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# ── NEW: Prometheus metrics ───────────────────────────────────────
try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server, generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response as FastAPIResponse
    PROMETHEUS_AVAILABLE = True

    PREDICTIONS_TOTAL    = Counter("sheep_predictions_total",    "Total predictions made",           ["breed", "confidence_bucket", "pipeline"])
    PREDICTION_LATENCY   = Histogram("sheep_prediction_latency_seconds", "Prediction latency in seconds", buckets=[.05,.1,.25,.5,1,2,5])
    CONFIDENCE_HISTOGRAM = Histogram("sheep_prediction_confidence",      "Model confidence scores",        buckets=[.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
    ACTIVE_REQUESTS      = Gauge("sheep_active_requests",        "Currently active requests")
    DIAGNOSE_TOTAL       = Counter("sheep_diagnose_total",       "Total diagnose calls",              ["status"])
    CUSTOM_MODEL_HITS    = Counter("sheep_custom_model_hits",    "Custom model usage",                ["result"])  # hit / fallback
    DATA_DRIFT_SCORE     = Gauge("sheep_data_drift_score",       "Rolling avg confidence (drift proxy)")
    _confidence_window   = []

    def _update_drift(conf: float):
        _confidence_window.append(conf)
        if len(_confidence_window) > 100:
            _confidence_window.pop(0)
        DATA_DRIFT_SCORE.set(sum(_confidence_window) / len(_confidence_window))

except ImportError:
    PROMETHEUS_AVAILABLE = False
    def _update_drift(conf): pass

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("sheep")

# ── NEW: MLflow experiment tracking ──────────────────────────────
try:
    import mlflow
    mlflow.set_experiment("sheep-breed-classifier")
    MLFLOW_AVAILABLE = True
    log.info("✓ MLflow tracking enabled")
except ImportError:
    MLFLOW_AVAILABLE = False
    log.warning("✗ MLflow not installed — experiment tracking disabled")

# ── Constants ─────────────────────────────────────────────────────
BREEDS       = ["Lohi", "Kajli", "Lari"]
MAX_UPLOAD   = 15 * 1024 * 1024
EID_MONTHS   = {"July", "August", "September", "October"}
VALID_BREEDS = {"Lohi", "Kajli", "Lari"}
VALID_AGES   = {"lamb", "young", "adult", "old"}
VALID_HEALTH = {"healthy", "weak", "diseased"}

BREED_RATE   = {"Lohi": 650, "Kajli": 700, "Lari": 600}
AGE_FACTOR   = {"lamb": 1.20, "young": 1.10, "adult": 1.00, "old": 0.75}
HEALTH_FACTOR= {"healthy": 1.00, "weak": 0.60, "diseased": 0.30}
EID_FACTOR   = 1.60
BREED_META   = {"Lohi": ("Sindh/Punjab","Wool, Meat"),
                "Kajli": ("Sindh/Punjab","Meat, Breeding"),
                "Lari": ("Sindh (coastal)","Meat")}

DISEASE_DB = {
    "Foot Rot":           {"symptoms":{"limping":0.8,"skin lesions":0.4,"weakness":0.2},          "severity":"Moderate", "actions":["Clean hoof area","Zinc sulfate footbath 2-3x/week","Keep on dry ground","Isolate infected animals","Consult vet if worsening"],           "description":"Bacterial hoof infection causing painful lameness."},
    "Sheep Pox":          {"symptoms":{"fever":0.7,"skin lesions":0.9,"loss of appetite":0.5,"nasal discharge":0.3}, "severity":"High",     "actions":["Isolate immediately","Apply antiseptic to lesions","Vaccinate healthy flock","Contact vet for antiviral treatment"],  "description":"Highly contagious viral skin disease. Progresses to pustules over 2 weeks."},
    "Pneumonia":          {"symptoms":{"fever":0.6,"coughing":0.9,"nasal discharge":0.8,"weakness":0.4},            "severity":"High",     "actions":["Provide warm shelter","Antibiotic treatment (Vet)","Improve ventilation","Reduce overcrowding"],                        "description":"Respiratory infection from stress or dust. Watch for specific breathing patterns."},
    "Enterotoxemia":      {"symptoms":{"diarrhea":0.6,"swollen belly":0.5,"weakness":0.8,"loss of appetite":0.4},   "severity":"Critical", "actions":["Reduce grain intake immediately","CD&T Vaccination","Immediate Vet attention","Provide oral electrolytes"],             "description":"Overeating disease caused by Clostridium. Rapidly fatal if untreated."},
    "Parasite Infestation":{"symptoms":{"diarrhea":0.5,"weight loss":0.9,"wool loss":0.4,"weakness":0.6},           "severity":"Moderate", "actions":["Deworming schedule","Rotate pastures","Check eye membranes (FAMACHA)","Fecal egg count test"],                          "description":"Worm infestation. Manage with deworming. Watch for parasite life cycle markers."},
    "Mastitis":           {"symptoms":{"udder swelling":1.0,"fever":0.4,"loss of appetite":0.3},                   "severity":"High",     "actions":["Keep bedding clean/dry","Hot compresses on udder","Intramammary antibiotics","Strip affected quarters frequently"],     "description":"Mammary gland inflammation, often bacterial."},
    "Bloat":              {"symptoms":{"swollen belly":1.0,"loss of appetite":0.6,"weakness":0.4},                  "severity":"Critical", "actions":["Walk sheep slowly","Anti-bloat drench","Avoid wet lush clover/alfalfa","Stomach tube if severe"],                      "description":"Gas buildup in rumen. Life-threatening if untreated quickly."},
    "Anthrax":            {"symptoms":{"fever":0.9,"weakness":0.7,"loss of appetite":0.5},                          "severity":"Critical", "actions":["Contact authorities immediately","Do NOT open carcass","Vaccinate entire flock","Quarantine affected area"],            "description":"Rare but fatal bacterial disease. Sudden death common. Report immediately."},
    "Scrapie":            {"symptoms":{"weakness":0.5,"wool loss":0.6,"weight loss":0.7},                           "severity":"Critical", "actions":["Report to animal health authorities","Cull affected animals","Genetic testing for flock","Do not breed from affected lines"], "description":"Progressive neurological disease. No treatment available."},
}

SYMPTOM_EXPANSIONS = {
    "fever":            "elevated body temperature hyperthermia pyrexia febrile condition",
    "coughing":         "persistent respiratory cough hacking productive cough dyspnea",
    "nasal discharge":  "nasal mucus secretion rhinitis mucopurulent discharge",
    "diarrhea":         "loose watery feces diarrhea enteritis gastrointestinal upset",
    "swollen belly":    "abdominal distension bloating rumen tympany gas accumulation",
    "loss of appetite": "anorexia reduced feed intake inappetence",
    "weakness":         "lethargy weakness depression ataxia reduced activity",
    "limping":          "lameness hoof pain claudication difficulty walking",
    "skin lesions":     "dermal lesions pustules papules skin eruptions vesicles",
    "wool loss":        "alopecia fleece loss wool break dermatitis",
    "udder swelling":   "mastitis mammary gland inflammation udder edema",
    "weight loss":      "cachexia progressive weight loss emaciation body condition loss",
}

# ── CLIP prompts ──────────────────────────────────────────────────
BREED_PROMPTS = {
    "Lohi":  ["a Lohi sheep with light tan brown woolly face from Punjab Pakistan",
               "Pakistani Lohi sheep breed with tan pinkish face and thick white wool body",
               "a wool and meat sheep from Punjab Pakistan with light brownish tan facial coloring",
               "Lohi breed sheep with characteristic tan face and fine wool fleece from Pakistani Punjab"],
    "Kajli": ["a Kajli sheep with bright white face and Roman nose from Sindh Pakistan",
               "Pakistani Kajli sheep with pure white face long Roman nose and tall long legged body",
               "a white faced sheep from Sindh Punjab Pakistan with distinctive Roman curved nose profile",
               "Kajli breed sheep with white face dark eye rings and tall elegant build from Pakistan"],
    "Lari":  ["a Lari sheep with faded brown coastal face from Sindh Pakistan",
               "Pakistani Lari sheep also called Jhari with desaturated light brown face from coastal Sindh",
               "a coastal Sindh sheep breed with faded brownish greyish face and heavy body frame",
               "Lari Jhari breed sheep with characteristic faded coastal coloring from southern Sindh Pakistan"],
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

# ── AI model globals ──────────────────────────────────────────────
YOLO_MODEL = CLIP_MODEL = CLIP_MODEL_2 = CLIP_PREPROCESS = CLIP_PREPROCESS_2 = VET_MODEL = None
DEVICE = CLIP_DEVICE = "cpu"
VET_DEVICE = -1

# ── NEW: Custom model globals ─────────────────────────────────────
CUSTOM_MODEL      = None   # ONNX fine-tuned EfficientNetV2
CUSTOM_PREPROCESS = None   # Albumentations pipeline
CUSTOM_THRESHOLD  = 0.75   # minimum confidence to trust custom model

try:
    import torch
    if torch.cuda.is_available():
        gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_gb >= 8:
            DEVICE = CLIP_DEVICE = "cuda"; VET_DEVICE = 0
        elif gpu_gb >= 6:
            DEVICE = CLIP_DEVICE = "cuda"
        log.info(f"GPU: {gpu_gb:.1f}GB — {'full' if gpu_gb>=8 else 'partial'} GPU mode")
    else:
        log.info("No GPU — using CPU")
except ImportError:
    log.warning("PyTorch not installed — AI disabled, pixel fallback active")

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
        log.info("✓ CLIP ViT-L/14 (secondary)")
    except Exception as e:
        log.warning(f"✗ CLIP secondary: {e}")
except Exception as e:
    log.warning(f"✗ CLIP: {e}")

try:
    from transformers import pipeline as hf_pipeline
    VET_MODEL = hf_pipeline("zero-shot-classification",
        model="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb", device=VET_DEVICE)
    log.info(f"✓ BioBERT (device={VET_DEVICE})")
except Exception as e:
    log.warning(f"✗ BioBERT: {e}")

# ── NEW: Load custom fine-tuned ONNX model (Layer 2) ─────────────
# To generate this model run: python train_custom.py
# It will output sheep_breed_classifier.onnx in the project root
try:
    import onnxruntime as ort

    # pick best ONNX execution provider automatically
    _providers = ort.get_available_providers()
    _ep = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in _providers \
          else ["CPUExecutionProvider"]

    CUSTOM_MODEL = ort.InferenceSession("sheep_breed_classifier.onnx", providers=_ep)
    log.info(f"✓ Custom EfficientNetV2 ONNX loaded (providers={_ep})")
except FileNotFoundError:
    log.warning("✗ Custom model (sheep_breed_classifier.onnx) not found — train first with train_custom.py")
except Exception as e:
    log.warning(f"✗ Custom model load failed: {e}")

# ── NEW: Albumentations preprocessing (matches training transforms)
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    CUSTOM_PREPROCESS = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    log.info("✓ Albumentations preprocessing ready")
except ImportError:
    log.warning("✗ Albumentations not installed — pip install albumentations")

# ── Pixel helpers (unchanged) ─────────────────────────────────────
def _rgb(img):
    a = np.array(img.resize((64,64)).convert("RGB"), dtype=np.float32)
    return a[:,:,0].mean(), a[:,:,1].mean(), a[:,:,2].mean()

def _region(img, x0, y0, x1, y1):
    w, h = img.size
    a = np.array(img.crop((int(w*x0),int(h*y0),int(w*x1),int(h*y1))).convert("RGB"), dtype=np.float32)
    return a[:,:,0].mean(), a[:,:,1].mean(), a[:,:,2].mean()

def _br(r,g,b):  return (r+g+b)/3
def _wht(r,g,b, t=185): return r>t and g>t and b>t
def _tan(r,g,b): return r>140 and r>g*1.08 and g>100 and b<155 and r<210
def _fade(r,g,b): return 90 < _br(r,g,b) < 175 and (max(r,g,b)-min(r,g,b)) < 55

def _pixel_fallback(img):
    fr, fg, fb = _region(img, .25, .0, .75, .30)
    w, h = img.size
    eye_c = (np.array(img.crop((int(w*.40),int(h*.05),int(w*.60),int(h*.20))).convert("RGB"),dtype=np.float32).mean()
           - (np.array(img.crop((int(w*.28),int(h*.12),int(w*.40),int(h*.24))).convert("RGB"),dtype=np.float32).mean()
            + np.array(img.crop((int(w*.60),int(h*.12),int(w*.72),int(h*.24))).convert("RGB"),dtype=np.float32).mean()) / 2)
    wool = float(np.array(img.crop((int(w*.15),int(h*.60),int(w*.85),int(h*.85))).convert("L"),dtype=np.float32).std())
    body_fill = np.array(img.crop((int(w*.1),int(h*.55),int(w*.9),int(h*.90))).convert("RGB"),dtype=np.float32).mean()

    if _wht(fr,fg,fb,175) and eye_c>18:    breed, conf, reason = "Kajli", 0.87, "White face with dark eye-ring contrast"
    elif _wht(fr,fg,fb,178):               breed, conf, reason = "Kajli", 0.76, "Pure white face — Kajli type"
    elif _tan(fr,fg,fb) and wool>38:        breed, conf, reason = "Lohi",  0.85, "Tan face + wool texture — Lohi"
    elif _tan(fr,fg,fb):                    breed, conf, reason = "Lohi",  0.75, "Light tan face — Lohi phenotype"
    elif _fade(fr,fg,fb) and body_fill>145: breed, conf, reason = "Lari",  0.78, "Faded coastal face + heavy body — Lari"
    else:                                   breed, conf, reason = "Lari",  0.52, "Faded light-brown face — Lari coastal"

    ob = _br(*_rgb(img))
    age, ac   = ("lamb",0.55) if ob>190 else ("young",0.55) if ob>155 else ("adult",0.60) if ob>110 else ("old",0.52)
    r,g,b     = _rgb(img)
    sat       = float(max(r,g,b)-min(r,g,b))
    health,hc = ("healthy",0.65) if sat>55 else ("weak",0.58) if sat>25 else ("diseased",0.52)

    rem = [b for b in BREEDS if b != breed]
    sp  = (1 - conf) / max(len(rem), 1)
    return {
        "is_sheep": True, "breed": breed, "breed_confidence": round(conf,3),
        "top_breeds": [{"breed":breed,"probability":round(conf,3)}] + [{"breed":b,"probability":round(sp,3)} for b in rem[:2]],
        "age": age, "age_confidence": round(ac,3), "health": health, "health_confidence": round(hc,3),
        "yolo_crop": False, "bcs": 3 if health=="healthy" else (2 if health=="weak" else 1),
        "wool": "Fine" if breed=="Lohi" else "Medium", "use": BREED_META[breed][1],
        "pipeline": "Pixel fallback", "reason": reason, "_price_breed": breed,
    }

# ── CLIP scoring (unchanged) ──────────────────────────────────────
def _clip_score(model, preprocess, img_pil, prompt_dict, device):
    import torch, clip as clip_lib
    img_t = preprocess(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        img_f = model.encode_image(img_t)
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        return {
            label: sum(
                (img_f @ (model.encode_text(clip_lib.tokenize([p]).to(device)) /
                          model.encode_text(clip_lib.tokenize([p]).to(device)).norm(dim=-1,keepdim=True)).T).item()
                for p in prompts
            ) / len(prompts)
            for label, prompts in prompt_dict.items()
        }

def _softmax_probs(scores, temp=5.0):
    import math
    exp_v = {k: math.exp(v*temp) for k,v in scores.items()}
    total = sum(exp_v.values())
    return {k: v/total for k,v in exp_v.items()}

def _confidence_from_scores(scores):
    vals = list(scores.values())
    mean = sum(vals) / len(vals)
    return round(min(0.92, max(0.55, max(vals) / (mean + 1e-6) * 0.5)), 3)

# ── NEW: Custom model inference (Layer 2) ────────────────────────
def _softmax_np(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def _custom_classify(img_pil: Image.Image) -> dict | None:
    """
    Run custom fine-tuned EfficientNetV2 ONNX model.
    Returns result dict if confidence >= CUSTOM_THRESHOLD, else None.
    Falls back gracefully — never raises.
    """
    if CUSTOM_MODEL is None or CUSTOM_PREPROCESS is None:
        return None
    try:
        img_np = np.array(img_pil.convert("RGB").resize((224, 224)), dtype=np.float32)
        augmented = CUSTOM_PREPROCESS(image=img_np)
        arr = augmented["image"]                          # HWC float32 normalized
        arr = np.transpose(arr, (2, 0, 1))               # CHW
        arr = np.expand_dims(arr, 0).astype(np.float32)  # NCHW

        input_name = CUSTOM_MODEL.get_inputs()[0].name
        raw = CUSTOM_MODEL.run(None, {input_name: arr})[0][0]  # shape: (3,)
        probs = _softmax_np(raw)

        top_idx  = int(np.argmax(probs))
        top_conf = float(probs[top_idx])

        if PROMETHEUS_AVAILABLE:
            CUSTOM_MODEL_HITS.labels(result="hit" if top_conf >= CUSTOM_THRESHOLD else "fallback").inc()

        if top_conf < CUSTOM_THRESHOLD:
            return None  # not confident enough — let CLIP handle it

        top_breeds = [
            {"breed": BREEDS[i], "probability": round(float(p), 3)}
            for i, p in sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
        ]
        return {
            "breed":            BREEDS[top_idx],
            "breed_confidence": round(top_conf, 3),
            "top_breeds":       top_breeds,
            "pipeline":         f"Custom EfficientNetV2-ONNX [conf:{round(top_conf,3)}]",
            "reason":           f"Fine-tuned model — {BREEDS[top_idx]} with {round(top_conf*100,1)}% confidence",
        }
    except Exception as e:
        log.warning(f"Custom model inference failed: {e}")
        if PROMETHEUS_AVAILABLE:
            CUSTOM_MODEL_HITS.labels(result="error").inc()
        return None

# ── Classification (Layer 2 inserted, everything else unchanged) ──
def classify(img: Image.Image) -> dict:
    # ── Layer 2: Custom fine-tuned model (NEW — tried first after YOLO crop) ──
    # This block is skipped entirely if model file is missing
    # No changes to pixel fallback or CLIP below

    if CLIP_MODEL is None:
        return _pixel_fallback(img)
    try:
        import torch

        # Layer 1: YOLO crop (unchanged)
        sheep_img, yolo_crop = img, False
        if YOLO_MODEL is not None:
            try:
                results = YOLO_MODEL(img, verbose=False)
                best = max(
                    ((box, float(box.conf[0])) for r in results for box in r.boxes
                     if YOLO_MODEL.names.get(int(box.cls[0].item())) == "sheep"),
                    key=lambda x: x[1], default=(None, 0)
                )
                if best[0] is not None:
                    b = best[0].xyxy[0].tolist()
                    w, h = img.size
                    p = 0.10
                    sheep_img = img.crop((max(0,int(b[0]-w*p)), max(0,int(b[1]-h*p)),
                                         min(w,int(b[2]+w*p)), min(h,int(b[3]+h*p))))
                    yolo_crop = True
                else:
                    return {"is_sheep": False, "error": "no_sheep_detected",
                            "breed":"Unknown","breed_confidence":0.0,"top_breeds":[],
                            "age":"unknown","age_confidence":0.0,"health":"unknown","health_confidence":0.0,
                            "yolo_crop":False,"bcs":0,"wool":"—","use":"—",
                            "pipeline":"YOLOv8x — no sheep detected","reason":"No sheep found","_price_breed":"Lohi"}
            except Exception as e:
                log.warning(f"YOLO failed: {e}")

        # ── Layer 2: Custom model — NEW insertion point ───────────
        custom_result = _custom_classify(sheep_img)
        if custom_result is not None:
            # Custom model is confident — skip CLIP entirely, run age/health via CLIP still
            sw, sh   = sheep_img.size
            body_img = sheep_img.resize((336,336), Image.LANCZOS)
            age_scores    = _clip_score(CLIP_MODEL, CLIP_PREPROCESS, body_img, AGE_PROMPTS, CLIP_DEVICE)
            health_scores = _clip_score(CLIP_MODEL, CLIP_PREPROCESS, body_img, HEALTH_PROMPTS, CLIP_DEVICE)
            age    = max(age_scores, key=age_scores.get)
            health = max(health_scores, key=health_scores.get)
            breed  = custom_result["breed"]

            return _validate_classify({
                "is_sheep":         True,
                "breed":            breed,
                "breed_confidence": custom_result["breed_confidence"],
                "top_breeds":       custom_result["top_breeds"],
                "age":              age,
                "age_confidence":   _confidence_from_scores(age_scores),
                "health":           health,
                "health_confidence":_confidence_from_scores(health_scores),
                "yolo_crop":        yolo_crop,
                "bcs":              3 if health=="healthy" else (2 if health=="weak" else 1),
                "wool":             "Fine" if breed=="Lohi" else "Medium",
                "use":              BREED_META.get(breed,("Sindh","Meat"))[1],
                "pipeline":         custom_result["pipeline"],
                "reason":           custom_result["reason"],
                "_price_breed":     breed,
            })

        # ── Layer 3: CLIP ensemble (unchanged) ────────────────────
        sw, sh = sheep_img.size
        face_img = sheep_img.crop((int(sw*.05),0,int(sw*.95),int(sh*.38))).resize((336,336), Image.LANCZOS)
        body_img = sheep_img.resize((336,336), Image.LANCZOS)

        s1 = _clip_score(CLIP_MODEL, CLIP_PREPROCESS, face_img, BREED_PROMPTS, CLIP_DEVICE)
        s2 = (_clip_score(CLIP_MODEL_2, CLIP_PREPROCESS_2, sheep_img.crop((int(sw*.05),0,int(sw*.95),int(sh*.38))),
                          BREED_PROMPTS, CLIP_DEVICE)
              if CLIP_MODEL_2 else {})
        px  = _pixel_fallback(img)
        ps  = {b: 0.05 for b in BREEDS}; ps[px["breed"]] = px["breed_confidence"] * 0.35

        def normalise(scores):
            mn, mx = min(scores.values()), max(scores.values())
            rng = max(mx - mn, 1e-6)
            return {k: (v-mn)/rng for k,v in scores.items()}

        ensemble = {}
        if s1: [ensemble.update({b: ensemble.get(b,0) + normalise(s1)[b]*0.55}) for b in BREEDS]
        if s2: [ensemble.update({b: ensemble.get(b,0) + normalise(s2)[b]*0.30}) for b in BREEDS]
        [ensemble.update({b: ensemble.get(b,0) + ps.get(b,0)*0.15}) for b in BREEDS]

        probs     = _softmax_probs(ensemble)
        breed     = max(probs, key=probs.get)
        breed_c   = round(probs[breed], 3)

        if breed_c < 0.35:
            breed, breed_c = px["breed"], px["breed_confidence"]
            pipeline = f"Pixel fallback [low conf:{breed_c}]"
        else:
            pipeline = f"CLIP-L14@336 ensemble [{CLIP_DEVICE}] conf:{breed_c}"

        age_scores    = _clip_score(CLIP_MODEL, CLIP_PREPROCESS, body_img, AGE_PROMPTS, CLIP_DEVICE)
        health_scores = _clip_score(CLIP_MODEL, CLIP_PREPROCESS, body_img, HEALTH_PROMPTS, CLIP_DEVICE)
        age    = max(age_scores, key=age_scores.get)
        health = max(health_scores, key=health_scores.get)

        REASON = {"Lohi":  "CLIP detected tan/brown woolly face — Lohi (Punjab/Sindh wool breed)",
                  "Kajli": "CLIP detected bright white face with Roman nose — Kajli",
                  "Lari":  "CLIP detected faded coastal colouring with heavy body — Lari (coastal Sindh)"}

        return _validate_classify({
            "is_sheep": True, "breed": breed, "breed_confidence": breed_c,
            "top_breeds": [{"breed":b,"probability":round(p,3)}
                           for b,p in sorted(probs.items(), key=lambda x:x[1], reverse=True)[:3]],
            "age": age, "age_confidence": _confidence_from_scores(age_scores),
            "health": health, "health_confidence": _confidence_from_scores(health_scores),
            "yolo_crop": yolo_crop, "bcs": 3 if health=="healthy" else (2 if health=="weak" else 1),
            "wool": "Fine" if breed=="Lohi" else "Medium", "use": BREED_META.get(breed,("Sindh","Meat"))[1],
            "pipeline": pipeline, "reason": REASON.get(breed,""), "_price_breed": breed,
        })
    except Exception as e:
        log.warning(f"AI classify failed: {e} — pixel fallback")
        return _pixel_fallback(img)

# ── Diagnosis (unchanged) ─────────────────────────────────────────
def _dict_diagnosis(selected):
    if not selected:
        return {"predicted_diseases":[], "health_score":100, "status":"Healthy", "diet":_diet_advice(selected)}
    preds = []
    for dis, data in DISEASE_DB.items():
        matched = {s: data["symptoms"][s] for s in selected if s in data["symptoms"]}
        if not matched: continue
        p = min(0.98, (sum(matched.values()) / sum(data["symptoms"].values()))
                     * (len(matched) / len(data["symptoms"])))
        preds.append({"disease":dis, "probability":round(p,2), "severity":data["severity"],
                      "actions":data["actions"], "description":data["description"]})
    preds.sort(key=lambda x: x["probability"], reverse=True)
    hs = max(0, min(100, 100 - len(selected)*12))
    return {"predicted_diseases":preds, "health_score":hs,
            "status":"High Risk" if hs<40 else "Moderate Risk" if hs<75 else "Healthy",
            "diet":_diet_advice(selected)}

def diagnose(selected: list) -> dict:
    if not selected:
        return {"predicted_diseases":[], "health_score":100, "status":"Healthy", "diet":_diet_advice([])}
    if VET_MODEL is None:
        return _dict_diagnosis(selected)
    try:
        clinical = (f"Ovine patient presenting with {'acute' if len(selected)<3 else 'chronic multi-symptom'} presentation: "
                    f"{', '.join(SYMPTOM_EXPANSIONS.get(s,s) for s in selected)}. "
                    "Livestock veterinary assessment required for Pakistani sheep breeds.")
        ai_res   = VET_MODEL(clinical, list(DISEASE_DB.keys()), multi_label=True)
        ai_map   = dict(zip(ai_res["labels"], ai_res["scores"]))
        dict_map = {d["disease"]: d["probability"] for d in _dict_diagnosis(selected)["predicted_diseases"]}

        preds = [
            {"disease":dis, "probability":round(min(0.98, ai_map.get(dis,0)*0.70 + dict_map.get(dis,0)*0.30),2),
             "severity":data["severity"], "actions":data["actions"], "description":data["description"]}
            for dis, data in DISEASE_DB.items()
            if ai_map.get(dis,0)*0.70 + dict_map.get(dis,0)*0.30 > 0.03
        ]
        preds.sort(key=lambda x: x["probability"], reverse=True)
        base = 100 - len(selected)*12
        if preds: base = int(base*(1 - preds[0]["probability"]*0.4))
        hs = max(0, min(100, base))
        return _validate_diagnosis({"predicted_diseases":preds, "health_score":hs,
            "status":"High Risk" if hs<40 else "Moderate Risk" if hs<75 else "Healthy",
            "diet":_diet_advice(selected)}, selected)
    except Exception as e:
        log.warning(f"AI diagnosis failed: {e}")
        return _dict_diagnosis(selected)

# ── Utilities (unchanged) ─────────────────────────────────────────
def _diet_advice(s):
    if any(x in s for x in ("diarrhea","swollen belly")):
        return ["Dry hay only","Reduce grain intake","Increase hydration","Electrolyte solution"]
    if any(x in s for x in ("weakness","weight loss")):
        return ["Protein supplements","Cottonseed cake","Fresh grass","Clean water","Vitamin B complex"]
    return ["Green fodder","Wheat straw","Corn feed","Mineral salt block","Fresh clean water"]

def image_quality(img):
    try:
        gray = np.array(img.convert("L"), dtype=np.float32)
        lum  = float(gray.mean())
        if CV2_AVAILABLE:
            sharp = float(cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var())
        else:
            sharp = float(np.var(gray[1:]-gray[:-1]) + np.var(gray[:,1:]-gray[:,:-1]))
        return {"is_blurry":sharp<80, "is_overexposed":lum>220, "is_underexposed":lum<35,
                "sharpness":round(sharp,1), "luminance":round(lum,1)}
    except:
        return {"is_blurry":False,"is_overexposed":False,"is_underexposed":False,"sharpness":0,"luminance":128}

def price(breed, age, health, wkg):
    is_eid = datetime.datetime.now().strftime("%B") in EID_MONTHS
    br = BREED_RATE.get(breed, BREED_RATE["Lohi"])
    af = AGE_FACTOR.get(age, 1.0)
    hf = HEALTH_FACTOR.get(health, 1.0)
    ef = EID_FACTOR if is_eid else 1.0
    rpk = br * af * hf * ef
    est = int(wkg * rpk)
    return {"weight_kg":round(wkg,1),"base_rate":br,"age_factor":af,"health_factor":hf,
            "eid_factor":round(ef,2),"rate_per_kg":round(rpk),"estimated":est,
            "min":int(est*.92),"max":int(est*1.08),"eid":is_eid,
            "display":f"PKR {int(est*.92):,} – {int(est*1.08):,}",
            "formula":f"{wkg:.1f} kg × {round(rpk):,} PKR/kg"}

def market_prices():
    is_eid = datetime.datetime.now().strftime("%B") in EID_MONTHS
    rows = []
    for breed, base in BREED_RATE.items():
        row = {"breed":breed,"eid":is_eid,"region":BREED_META[breed][0],"uses":BREED_META[breed][1],"base_rate_kg":base}
        for age, af in AGE_FACTOR.items():
            for hlth, hf in HEALTH_FACTOR.items():
                ef  = EID_FACTOR if is_eid else 1.0
                r   = round(base*af*hf*ef)
                ref = int(40*r)
                row[f"{age}_{hlth}_rate"] = r
                row[f"{age}_{hlth}_min"]  = int(ref*.92)
                row[f"{age}_{hlth}_max"]  = int(ref*1.08)
        rows.append(row)
    return {"prices":rows,"eid":is_eid,"ref_weight":40,"updated":datetime.datetime.now().isoformat()}

# ── Output validators (unchanged) ─────────────────────────────────
def _validate_classify(r):
    if r.get("breed") not in VALID_BREEDS:  r["breed"]  = "Lari"
    if r.get("age")   not in VALID_AGES:    r["age"]    = "adult"
    if r.get("health")not in VALID_HEALTH:  r["health"] = "healthy"
    for k in ("breed_confidence","age_confidence","health_confidence"):
        r[k] = round(min(1.0, max(0.0, float(r.get(k,0.5)))), 3)
    r["bcs"] = max(1, min(5, int(r.get("bcs",3))))
    if not r.get("top_breeds"):
        b, c  = r["breed"], r["breed_confidence"]
        rem   = [x for x in BREEDS if x!=b]
        sp    = (1-c)/max(len(rem),1)
        r["top_breeds"] = [{"breed":b,"probability":round(c,3)}] + [{"breed":x,"probability":round(sp,3)} for x in rem[:2]]
    return r

def _validate_diagnosis(r, selected):
    r["health_score"] = max(0, min(100, int(r.get("health_score",100))))
    if r.get("status") not in {"Healthy","Moderate Risk","High Risk"}: r["status"] = "Moderate Risk"
    if not isinstance(r.get("diet"), list): r["diet"] = _diet_advice(selected)
    clean = []
    for d in r.get("predicted_diseases",[]):
        if not isinstance(d,dict) or "disease" not in d or "probability" not in d: continue
        d["probability"] = round(min(0.98, max(0.0, float(d.get("probability",0)))), 2)
        d.setdefault("severity","Moderate"); d.setdefault("actions",[]); d.setdefault("description","")
        clean.append(d)
    r["predicted_diseases"] = clean
    return r

# ── Embedded Frontend HTML ────────────────────────────────────────
# NOTE: The full frontend is inlined here so this is a single-file deployment.
# No external frontend.html file is needed.
FRONTEND_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Sheppy</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
:root {
  --font-display: 'Bebas Neue', 'Impact', sans-serif;
  --font-body:    'Space Grotesk', 'Helvetica Neue', sans-serif;
  --font-mono:    'Space Mono', 'Courier New', monospace;
}
</style>
<style>
:root{--bg:#09090a;--s1:#0f0e0b;--s2:#161410;--s3:#1c1a12;--bd:#2a2616;--or:#b8922a;--orl:#d4aa3f;--orb:#e8c55a;--ord:rgba(184,146,42,.18);--tx:#f0e4c4;--tm:#7a6a45;--td:#3a3320;--gr:#3dd68c;--re:#e05555;--ye:#e8a020;--dim:#c8b47a}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:var(--font-body);font-size:12.5px;font-weight:400;line-height:1.6;background:transparent;color:var(--tx);display:flex;height:100vh;overflow:hidden}
/* Sidebar */
.sidebar{width:185px;min-width:185px;background:linear-gradient(180deg,rgba(12,11,8,.92),rgba(9,9,10,.93));border-right:1px solid var(--bd);display:flex;flex-direction:column;backdrop-filter:blur(12px)}
.logo{display:flex;align-items:center;gap:10px;padding:14px 12px;border-bottom:1px solid var(--bd)}
.logo-icon{width:36px;height:36px;flex-shrink:0;cursor:pointer;filter:drop-shadow(0 0 6px rgba(184,146,42,.25));transition:filter .3s}
.logo-icon:hover{filter:drop-shadow(0 0 14px rgba(232,197,90,.6))}
.logo-text-block{display:flex;flex-direction:column;gap:0px}
.logo-main{font-size:14px;font-weight:400;font-family:var(--font-display);letter-spacing:.06em;background:linear-gradient(90deg,#7a5c1a,#d4aa3f,#e8c55a,#d4aa3f,#7a5c1a);background-size:300%;-webkit-background-clip:text;-webkit-text-fill-color:transparent;animation:gold-sweep 5s ease-in-out infinite;line-height:1}
.logo-sub{font-size:6.5px;font-weight:700;font-family:var(--font-body);letter-spacing:.32em;color:var(--tm);text-transform:uppercase;margin-top:2px}
.nav{padding:7px 6px;flex:1;display:flex;flex-direction:column;gap:2px}
.ni{display:flex;align-items:center;gap:7px;padding:0 9px;height:36px;border-radius:5px;cursor:pointer;font-size:11.5px;font-weight:500;font-family:var(--font-body);letter-spacing:.01em;color:var(--tm);transition:.15s;border:1px solid transparent;user-select:none}
.ni:hover{background:var(--s2);color:var(--tx)}
.ni.active{background:linear-gradient(90deg,rgba(184,146,42,.22),rgba(184,146,42,.08));border-color:rgba(184,146,42,.5);color:var(--orl);font-weight:600}
.lang-box{margin:6px;border:1px solid var(--bd);border-radius:6px;padding:8px;background:var(--s2)}
.lt{font-size:8px;font-weight:700;font-family:var(--font-body);color:var(--tm);text-transform:uppercase;letter-spacing:.24em;margin-bottom:6px}
.lo{padding:5px 7px;border-radius:4px;font-size:11px;font-weight:500;font-family:var(--font-body);cursor:pointer;color:var(--tm);transition:.15s}
.lo:hover{color:var(--tx)}.lo.active{background:var(--ord);color:var(--orl);font-weight:600}
/* Main */
.main{flex:1;display:flex;flex-direction:column;overflow:hidden;background:rgba(9,9,10,.12)}
.topbar{display:flex;align-items:center;padding:0 18px;height:48px;border-bottom:1px solid var(--bd);background:rgba(12,11,8,.88);backdrop-filter:blur(14px)}
.topbar h1{font-size:18px;font-weight:400;font-family:var(--font-display);letter-spacing:.08em;color:var(--tx);line-height:1}
.content{flex:1;display:flex;overflow:hidden}
.center{flex:1;padding:15px;overflow-y:auto;display:flex;flex-direction:column;gap:11px}
.center::-webkit-scrollbar{width:3px}.center::-webkit-scrollbar-thumb{background:var(--bd);border-radius:2px}
.panel{background:rgba(12,11,8,.88);border:1px solid var(--bd);border-radius:10px;padding:14px;backdrop-filter:blur(10px)}
.ph{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;min-height:36px}
.pt{font-size:12px;font-weight:600;font-family:var(--font-body);letter-spacing:.01em;color:var(--tx)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:11px}
/* Buttons */
.btn{background:linear-gradient(135deg,#b8922a,#d4aa3f);color:#0a0a0a;border:none;padding:7px 16px;border-radius:6px;font-size:13px;font-weight:400;font-family:var(--font-display);letter-spacing:.06em;cursor:pointer;transition:.15s}
.btn:hover{background:linear-gradient(135deg,#d4aa3f,#e8c55a)}
.btn:disabled,.analyze-btn:disabled{background:var(--s3);color:var(--td);cursor:not-allowed}
.btn2{background:var(--s2);border:1px solid var(--bd);color:var(--tm);padding:7px 12px;border-radius:6px;font-size:10px;font-weight:600;font-family:var(--font-body);letter-spacing:.05em;cursor:pointer;transition:.15s}
.btn2:hover{border-color:var(--tm);color:var(--tx)}
.btn-danger{background:var(--s2);border:1px solid rgba(224,85,85,.4);color:var(--re);padding:7px 12px;border-radius:6px;font-size:10px;font-weight:600;font-family:var(--font-body);letter-spacing:.04em;cursor:pointer}
.ifield{width:100%;background:var(--s3);border:1px solid var(--bd);border-radius:6px;padding:7px 9px;font-size:11.5px;font-weight:400;color:var(--tx);font-family:var(--font-body);outline:none;transition:.15s}
.ifield:focus{border-color:var(--or)}
.fl{font-size:9px;color:var(--tm);font-weight:700;font-family:var(--font-body);text-transform:uppercase;letter-spacing:.18em;margin-bottom:5px;display:block}
.fg{margin-bottom:10px}
/* Upload */
.upload-zone{border:2px dashed var(--or);border-radius:10px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:7px;padding:24px 14px;cursor:pointer;transition:.2s;position:relative;min-height:195px;background:linear-gradient(160deg,rgba(184,146,42,.06),rgba(184,146,42,.02));overflow:hidden}
.upload-zone:hover,.upload-zone.drag{background:rgba(184,146,42,.1);border-color:#d4aa3f}
.upload-zone input{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
.uicon{width:46px;height:46px;border:2px solid var(--or);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:19px;color:var(--or)}
.upload-zone p{font-size:11.5px;font-weight:500;font-family:var(--font-body);color:var(--or)}
.upload-zone small{font-size:9.5px;font-weight:400;font-family:var(--font-body);color:var(--td)}
.prev-img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;border-radius:6px;display:none}
.prev-img.show{display:block}
.prev-badge{position:absolute;bottom:7px;left:7px;background:rgba(0,0,0,.75);color:var(--or);font-size:10px;padding:2px 7px;border-radius:4px;border:1px solid var(--or);display:none}
.prev-badge.show,.eid-banner.show,.eid-badge.show{display:flex}.eid-badge.show{display:block}
.result-area{position:relative;border-radius:8px;overflow:hidden;min-height:195px;background:var(--s2);display:flex;flex-direction:column}
.rph{flex:1;display:flex;align-items:center;justify-content:center;font-size:40px;color:var(--td)}
.rimg{width:100%;height:170px;object-fit:cover;border-radius:8px 8px 0 0;display:none}
.rov{position:absolute;top:7px;right:7px;background:rgba(12,12,12,.93);border:1px solid var(--bd);border-radius:8px;padding:9px 11px;min-width:138px;display:none;font-size:11px;font-family:var(--font-body);backdrop-filter:blur(4px)}
.or2{display:flex;justify-content:space-between;gap:10px;margin-bottom:3px}
.ol{color:var(--tm)}.ov{font-weight:600}
.hbadge{display:inline-flex;align-items:center;border-radius:4px;padding:2px 7px;font-size:9px;font-weight:600;font-family:var(--font-body);letter-spacing:.1em;text-transform:uppercase;margin-top:2px}
.hbadge.healthy{background:rgba(61,214,140,.12);border:1px solid rgba(61,214,140,.35);color:var(--gr)}
.hbadge.weak{background:rgba(232,160,32,.12);border:1px solid rgba(232,160,32,.35);color:var(--ye)}
.hbadge.diseased{background:rgba(224,85,85,.12);border:1px solid rgba(224,85,85,.35);color:var(--re)}
.analyze-btn{width:100%;background:linear-gradient(135deg,#b8922a,#d4aa3f);color:#0a0a0a;border:none;padding:10px;border-radius:0 0 7px 7px;font-size:16px;font-weight:400;font-family:var(--font-display);letter-spacing:.08em;cursor:pointer}
.cbar{margin-bottom:4px}.cbll{display:flex;justify-content:space-between;font-size:10px;color:var(--tm);margin-bottom:2px}
.cbt{height:4px;background:var(--s3);border-radius:2px;overflow:hidden}
.cbf{height:100%;border-radius:2px;background:linear-gradient(90deg,var(--or),var(--orl));transition:width .7s}
.cbf.top{background:linear-gradient(90deg,#d4aa3f,#e8c55a)}
.feat-grid{display:grid;grid-template-columns:1fr 1fr;gap:5px;margin-top:8px}
.fi{display:flex;align-items:center;gap:5px;font-size:11px;font-weight:500;font-family:var(--font-body);color:var(--tm)}
.fd{width:12px;height:12px;border-radius:50%;border:1.5px solid var(--or);flex-shrink:0;font-size:7px;color:var(--or);display:flex;align-items:center;justify-content:center}
.reason-box{background:var(--s3);border:1px solid var(--bd);border-radius:6px;padding:8px 10px;font-size:11px;font-weight:400;font-family:var(--font-body);color:var(--tm);margin-top:8px;line-height:1.6}
.reason-box strong{color:var(--or)}
/* Records */
.recs-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.rc{background:var(--s2);border:1px solid var(--bd);border-radius:8px;display:flex;align-items:center;gap:8px;padding:8px;position:relative;transition:.15s}
.rc:hover{border-color:var(--or)}
.rthumb{width:44px;height:44px;border-radius:6px;background:var(--s3);flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:16px;color:var(--td);overflow:hidden}
.rthumb img,.rlci img{width:100%;height:100%;object-fit:cover}
.rthumb img{border-radius:6px}
.rsb{width:15px;height:15px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:9px}
.rsb.good{background:rgba(61,214,140,.12);color:var(--gr)}.rsb.bad{background:rgba(224,85,85,.12);color:var(--re)}
.del-btn,.rlc-del{position:absolute;border-radius:50%;background:rgba(224,85,85,.85);border:none;color:#fff;cursor:pointer;display:flex;align-items:center;justify-content:center;opacity:0;transition:.15s;padding:0}
.del-btn{top:4px;right:4px;width:16px;height:16px;font-size:9px}
.rlc-del{top:5px;right:5px;width:20px;height:20px;font-size:11px;z-index:5}
.rc:hover .del-btn,.rlc:hover .rlc-del{opacity:1}
.rlg{display:grid;grid-template-columns:repeat(3,1fr);gap:9px}
.rlc{background:var(--s2);border:1px solid var(--bd);border-radius:8px;overflow:hidden;position:relative;transition:.15s}
.rlc:hover{border-color:var(--or)}
.rlci{width:100%;height:82px;background:var(--s3);display:flex;align-items:center;justify-content:center;font-size:26px;color:var(--td);overflow:hidden}
.rlcb{padding:8px}.rlcbr{font-size:11.5px;font-weight:600;font-family:var(--font-body);letter-spacing:.02em;margin-bottom:3px}
.rlcm{display:flex;justify-content:space-between;font-size:10px;font-weight:400;font-family:var(--font-body);color:var(--tm)}
.rlcp{font-size:11.5px;font-weight:700;font-family:var(--font-body);color:var(--or);margin-top:3px}
/* Market */
.mtable{width:100%;border-collapse:collapse;font-size:11.5px;font-family:var(--font-body)}
.mtable th{text-align:left;padding:7px 9px;background:var(--s2);color:var(--tm);font-weight:600;font-family:var(--font-body);font-size:9px;text-transform:uppercase;letter-spacing:.14em;border-bottom:1px solid var(--bd)}
.mtable td{padding:7px 9px;border-bottom:1px solid var(--bd)}
.mtable tr:hover td{background:var(--s2)}
.pr{color:var(--or);font-weight:700;font-family:var(--font-mono)}
.mkt-filter{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:11px}
.mft{background:var(--s2);border:1px solid var(--bd);border-radius:20px;padding:4px 11px;font-size:9.5px;font-weight:600;font-family:var(--font-body);letter-spacing:.08em;text-transform:uppercase;color:var(--tm);cursor:pointer;transition:.15s}
.mft:hover,.mft.active{background:var(--ord);border-color:var(--or);color:var(--or)}
.eid-banner{background:linear-gradient(90deg,rgba(184,146,42,.18),rgba(184,146,42,.08));border:1px solid rgba(184,146,42,.4);border-radius:8px;padding:9px 12px;font-size:11.5px;font-weight:500;font-family:var(--font-body);color:var(--orl);margin-bottom:11px;display:none;align-items:center;gap:7px}
.live-dot{width:6px;height:6px;border-radius:50%;background:var(--gr);display:inline-block;margin-right:4px;animation:lp 2s infinite}
/* Modal */
.modal-ov{position:fixed;inset:0;background:rgba(0,0,0,.75);display:flex;align-items:center;justify-content:center;z-index:1000;opacity:0;pointer-events:none;transition:.2s}
.modal-ov.open{opacity:1;pointer-events:all}
.modal{background:var(--s1);border:1px solid var(--bd);border-radius:10px;width:400px;max-width:95vw;max-height:90vh;overflow-y:auto;transform:translateY(12px);transition:.2s}
.modal-ov.open .modal{transform:translateY(0)}
.mhdr{display:flex;align-items:center;justify-content:space-between;padding:13px 15px;border-bottom:1px solid var(--bd);font-family:var(--font-body)}
.mcls{width:24px;height:24px;background:var(--s2);border:1px solid var(--bd);border-radius:5px;display:flex;align-items:center;justify-content:center;cursor:pointer;font-size:12px;color:var(--tm)}
.mcls:hover{border-color:var(--re);color:var(--re)}
.mbdy{padding:14px;display:flex;flex-direction:column;gap:11px}
.mftr{padding:11px 14px;border-top:1px solid var(--bd);display:flex;gap:7px;justify-content:flex-end}
.frow{display:grid;grid-template-columns:1fr 1fr;gap:9px}
.fi2{background:var(--s2);border:1px solid var(--bd);border-radius:6px;padding:7px 9px;font-size:11.5px;font-weight:400;font-family:var(--font-body);color:var(--tx);outline:none;transition:.15s;width:100%}
.fi2:focus,.fsel:focus{border-color:var(--or)}
.fsel{background:var(--s2);border:1px solid var(--bd);border-radius:6px;padding:7px 9px;font-size:11.5px;font-weight:400;font-family:var(--font-body);color:var(--tx);outline:none;transition:.15s;width:100%;cursor:pointer;appearance:none;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%23888'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 9px center}
.iub{border:2px dashed var(--bd);border-radius:8px;padding:14px;text-align:center;cursor:pointer;transition:.15s;position:relative;overflow:hidden}
.iub:hover{border-color:var(--or);background:var(--ord)}
.iub input{position:absolute;inset:0;opacity:0;cursor:pointer}
.ipt{width:100%;height:65px;object-fit:cover;border-radius:5px;display:none}
/* Right sidebar */
.rsidebar{width:272px;min-width:272px;border-left:1px solid var(--bd);overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:9px;background:rgba(12,11,8,.9);backdrop-filter:blur(12px)}
.rsidebar::-webkit-scrollbar{width:3px}.rsidebar::-webkit-scrollbar-thumb{background:var(--bd);border-radius:2px}
.mcard{background:linear-gradient(160deg,rgba(22,20,16,.9),rgba(28,26,18,.9));border:1px solid var(--bd);border-radius:10px;padding:14px;backdrop-filter:blur(8px)}
.mcard-title{font-size:9.5px;font-weight:700;font-family:var(--font-body);color:var(--tm);text-transform:uppercase;letter-spacing:.24em;margin-bottom:10px}
.mprice{font-size:24px;font-weight:700;background:linear-gradient(90deg,#d4aa3f,#e8c55a);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-family:var(--font-mono);line-height:1.2}
.mrow{display:flex;justify-content:space-between;align-items:center;padding:4px 0;border-bottom:1px solid var(--bd);font-size:11px}
.mrow:last-child{border:none}
.mkey{color:var(--tm);font-size:10.5px;font-weight:400;font-family:var(--font-body)}.mval{font-weight:600;font-size:10.5px;font-family:var(--font-body)}.mval.gr{color:var(--gr)}.mval.ye{color:var(--ye)}.mval.re{color:var(--re)}.mval.or{color:var(--or)}
.eid-badge{background:rgba(184,146,42,.15);border:1px solid rgba(184,146,42,.4);border-radius:5px;padding:4px 8px;font-size:10px;color:var(--or);margin-top:6px;display:none}
.price-placeholder{text-align:center;padding:14px 10px;color:var(--td);font-size:11px;font-weight:400;font-family:var(--font-body);border:1px dashed var(--bd);border-radius:7px}
.breakdown{background:var(--s3);border:1px solid var(--bd);border-radius:6px;padding:8px;margin-top:7px;font-size:11px}
.bd-title{font-weight:600;font-family:var(--font-body);color:var(--tm);margin-bottom:6px;font-size:9px;text-transform:uppercase;letter-spacing:.16em}
.formula-tag{font-size:10px;color:var(--or);margin-top:5px;padding:4px 7px;background:rgba(184,146,42,.08);border:1px solid var(--bd);border-radius:4px;font-family:var(--font-mono)}
.view{display:none}.view.active{display:flex;flex-direction:column;gap:12px}
/* Health */
.ha-panel{display:flex;flex-direction:column;gap:0}
.ha-title{font-size:28px;font-weight:400;font-family:var(--font-display);letter-spacing:.04em;margin-bottom:10px;line-height:1}
.ha-results-wrap{display:flex;gap:14px;align-items:flex-start;margin-top:14px}
.ha-results-left{flex:1;min-width:0}
.ha-results-right{width:236px;min-width:236px;display:flex;flex-direction:column;gap:8px}
.ha-gauge-block{background:#0a0905;border:1px solid var(--bd);border-radius:10px;padding:12px;text-align:center}
.sym-grid{display:grid;grid-template-columns:repeat(8,1fr);gap:5px;margin-bottom:6px}
.sym-grid2{display:grid;grid-template-columns:repeat(5,1fr);gap:5px;margin-bottom:12px}
.sb{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:4px;padding:10px 6px 8px;background:#0f0e08;border:1px solid #1e1b10;border-radius:8px;cursor:pointer;transition:border-color .2s,background .2s,box-shadow .2s;position:relative;min-height:76px;user-select:none}
.sb:hover{border-color:rgba(184,146,42,.4);background:#161209}
.sb.on{background:linear-gradient(160deg,rgba(184,146,42,.22),rgba(184,146,42,.06));border-color:#c9a030;border-width:1.5px;box-shadow:0 0 12px rgba(184,146,42,.25),inset 0 0 8px rgba(184,146,42,.08)}
.sb.on .sb-ico{color:#e8c55a;filter:drop-shadow(0 0 4px rgba(232,197,90,.6))}.sb.on .sb-lbl{color:#d4aa3f}
.sb-ck{position:absolute;top:3px;right:3px;width:14px;height:14px;border-radius:50%;background:linear-gradient(135deg,#c9a030,#e8c55a);display:none;align-items:center;justify-content:center;font-size:8px;color:#000;font-weight:900;box-shadow:0 0 6px rgba(200,165,48,.7)}
.sb.on .sb-ck{display:flex}
.sb-ico{width:30px;height:30px;color:rgba(180,155,90,.45);transition:color .2s,filter .2s;display:flex;align-items:center;justify-content:center;flex-shrink:0}
.sb-ico svg{width:100%;height:100%}
.sb-lbl{font-size:9px;font-weight:400;font-family:var(--font-body);color:rgba(120,100,50,.85);text-align:center;line-height:1.3;transition:color .2s}
.obs-i{width:100%;background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.09);border-radius:6px;padding:9px 13px;font-size:11.5px;font-weight:400;font-family:var(--font-body);color:var(--tx);outline:none;transition:.15s;margin-bottom:12px}
.obs-i:focus{border-color:var(--or)}.obs-i::placeholder{color:var(--td)}
.ha-btn{width:100%;background:linear-gradient(135deg,#c9a030 0%,#f0d060 45%,#c9a030 100%);color:#0a0a08;border:none;padding:13px;border-radius:7px;font-size:20px;font-weight:400;font-family:var(--font-display);letter-spacing:.1em;cursor:pointer;transition:.25s;box-shadow:0 2px 18px rgba(184,146,42,.45)}
.ha-btn:hover{box-shadow:0 4px 32px rgba(184,146,42,.7),0 0 0 1px rgba(220,180,60,.5);background:linear-gradient(135deg,#d4aa3f 0%,#f5d855 45%,#d4aa3f 100%)}
.sec-t{font-size:9.5px;font-weight:700;font-family:var(--font-body);color:var(--tm);text-transform:uppercase;letter-spacing:.22em;margin-bottom:12px}
.gauge-wrap{position:relative;width:148px;height:100px;margin:0 auto 6px}
.gauge-wrap svg{width:148px;height:100px;overflow:visible}
.gauge-center{position:absolute;bottom:2px;left:50%;transform:translateX(-50%);text-align:center;pointer-events:none}
.gn{font-size:38px;font-weight:700;font-family:var(--font-mono);line-height:1;color:#e8c55a;text-shadow:0 0 14px rgba(232,197,90,.7),0 0 28px rgba(184,146,42,.4)}
.gs{font-size:8px;font-weight:700;color:var(--tm);font-family:var(--font-body);text-transform:uppercase;letter-spacing:.28em;margin-top:2px}
.g-status{font-size:16px;font-weight:400;font-family:var(--font-display);letter-spacing:.08em;text-align:center;margin-top:5px;text-shadow:0 0 10px currentColor}
.dc{background:#0f0e08;border:1px solid #221f12;border-radius:8px;margin-bottom:8px;overflow:hidden;transition:border-color .15s,box-shadow .15s}
.dc:hover{border-color:rgba(184,146,42,.4);box-shadow:0 2px 12px rgba(0,0,0,.3)}
.dc-top{display:flex;align-items:center;gap:10px;padding:10px 12px}
.dc-n{width:22px;height:22px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;font-family:var(--font-mono);flex-shrink:0}
.dc-icon{width:40px;height:40px;border-radius:7px;display:flex;align-items:center;justify-content:center;flex-shrink:0}
.dc-icon svg{width:24px;height:24px}
.dc-body{flex:1;min-width:0}
.dc-name{font-size:13px;font-weight:600;font-family:var(--font-body);letter-spacing:.01em;margin-bottom:2px;color:var(--tx)}
.dc-desc{font-size:10px;font-weight:400;font-family:var(--font-body);color:var(--tm);line-height:1.55;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.dc-right{display:flex;flex-direction:column;align-items:flex-end;gap:4px;flex-shrink:0;margin-left:8px}
.dc-pct{font-size:22px;font-weight:700;font-family:var(--font-mono);line-height:1;text-shadow:0 0 8px currentColor}
.dc-sev{font-size:8px;font-weight:700;font-family:var(--font-body);letter-spacing:.1em;text-transform:uppercase;padding:2px 8px;border-radius:4px;white-space:nowrap}
.dc-sev.critical{background:rgba(224,85,85,.18);border:1px solid rgba(224,85,85,.5);color:#ff7070;text-shadow:0 0 6px rgba(224,85,85,.5)}
.dc-sev.high{background:rgba(232,160,32,.15);border:1px solid rgba(232,160,32,.45);color:#f5b030;text-shadow:0 0 6px rgba(232,160,32,.4)}
.dc-sev.moderate{background:rgba(61,214,140,.1);border:1px solid rgba(61,214,140,.35);color:#4ade80;text-shadow:0 0 6px rgba(61,214,140,.3)}
.dc-btns{display:flex;gap:5px;margin-top:3px}
.dc-tb{background:#1a1712;border:1px solid rgba(255,255,255,.08);border-radius:4px;padding:3px 9px;font-size:9px;font-weight:600;font-family:var(--font-body);letter-spacing:.06em;text-transform:uppercase;color:var(--tm);cursor:pointer;transition:.15s}
.dc-tb:hover,.dc-tb.on{background:var(--ord);border-color:var(--or);color:var(--orl)}
.dc-bar{height:3px;background:#1a1712;margin:0 13px}
.dc-bar-f{height:100%;border-radius:2px;transition:width 1s ease;box-shadow:0 0 6px currentColor}
.dc-exp{display:none;padding:9px 13px 11px;border-top:1px solid #221f12}.dc-exp.on{display:block}
.act-row{display:flex;flex-wrap:wrap;gap:5px}
.act-chip{background:rgba(184,146,42,.08);border:1px solid rgba(184,146,42,.2);border-radius:5px;padding:3px 9px;font-size:10px;font-weight:500;font-family:var(--font-body);color:var(--dim)}
.sev-alert{background:rgba(224,85,85,.1);border:1px solid rgba(224,85,85,.3);border-radius:5px;padding:5px 10px;font-size:10px;font-weight:500;font-family:var(--font-body);color:#ff7070;margin:4px 13px 0;display:flex;align-items:center;gap:5px}
.mini-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:6px}
.mc{background:#0f0e08;border:1px solid #1e1b10;border-radius:7px;display:flex;align-items:center;gap:8px;padding:7px 10px}
.mc-ico{width:20px;height:20px;flex-shrink:0;color:var(--tm);opacity:.65}.mc-ico svg{width:100%;height:100%}
.mc-name{font-size:10.5px;font-weight:500;font-family:var(--font-body);flex:1;color:var(--dim)}
.mc-pct{font-size:11px;font-weight:700;font-family:var(--font-mono)}
.hs-panel{background:#0a0905;border:1px solid #1e1b10;border-radius:8px;padding:11px}
.spark-svg{width:100%;height:72px;overflow:visible}
.diet-item{display:flex;align-items:center;gap:7px;padding:5px 0;border-bottom:1px solid #1e1b10;font-size:11px;font-weight:400;font-family:var(--font-body);line-height:1.5}
.diet-item:last-child{border:none}
.d-dot{width:7px;height:7px;border-radius:50%;background:#d4aa3f;flex-shrink:0;box-shadow:0 0 5px rgba(212,170,63,.6)}
.consult-wrap{background:#0a0905;border:1px solid #1e1b10;border-radius:8px;padding:12px;display:none}.consult-wrap.on{display:block}
.ct{font-size:11.5px;font-weight:600;font-family:var(--font-body);letter-spacing:.02em;color:var(--tx);margin-bottom:6px}
.cl{list-style:none;padding:0;margin:5px 0 10px}
.cl li{display:flex;align-items:flex-start;gap:5px;font-size:11px;font-weight:400;font-family:var(--font-body);color:var(--tm);line-height:1.5;padding:3px 0}
.cl li::before{content:"•";color:#d4aa3f;flex-shrink:0}
.c-btns{display:grid;grid-template-columns:1fr 1fr;gap:7px}
.btn-treat{background:#141209;border:1px solid rgba(61,214,140,.25);border-radius:6px;padding:10px;font-size:14px;font-weight:400;font-family:var(--font-display);letter-spacing:.06em;color:var(--gr);cursor:pointer;text-align:center;transition:.15s}
.btn-treat:hover{border-color:var(--gr);color:var(--gr);box-shadow:0 0 8px rgba(61,214,140,.2)}
.btn-vet{background:linear-gradient(135deg,#c9a030,#e8c55a);color:#0a0a08;border:none;padding:10px;border-radius:6px;font-size:14px;font-weight:400;font-family:var(--font-display);letter-spacing:.06em;cursor:pointer;text-align:center;box-shadow:0 2px 14px rgba(184,146,42,.5)}
.btn-vet:hover{background:linear-gradient(135deg,#d4aa3f,#f0d060);box-shadow:0 4px 16px rgba(184,146,42,.6)}
.disclaim{font-size:9px;font-weight:400;font-family:var(--font-body);color:#3a3320;text-align:center;margin-top:8px;line-height:1.6;letter-spacing:.02em;border-top:1px solid #1e1b10;padding-top:8px}
.spin{width:13px;height:13px;border:2px solid var(--bd);border-top-color:var(--or);border-radius:50%;animation:sp .7s linear infinite;display:inline-block;vertical-align:middle;margin-right:5px}
@keyframes sp{to{transform:rotate(360deg)}}
@keyframes lp{0%,100%{opacity:1}50%{opacity:.4}}
@keyframes dot-pop{0%{transform:scale(0);opacity:0}70%{transform:scale(1.3)}100%{transform:scale(1);opacity:1}}
@keyframes rise-from-earth{0%{opacity:0;transform:translateY(14px) scale(.98);filter:blur(4px)}55%{opacity:.85;filter:blur(.8px)}100%{opacity:1;transform:translateY(0) scale(1);filter:blur(0)}}
@keyframes brand-burn{0%{opacity:0;transform:translateX(-10px)}60%{opacity:.9;transform:translateX(1px)}100%{opacity:1;transform:translateX(0)}}
@keyframes earmark{0%{opacity:0;transform:translate(12px,-12px) scale(.84) rotate(6deg)}60%{transform:translate(-1px,1px) scale(1.03) rotate(-.6deg)}100%{opacity:1;transform:translate(0,0) scale(1) rotate(0)}}
@keyframes weight-stamp{0%{opacity:0;transform:scale(1.6) rotate(-3deg)}50%{transform:scale(.94) rotate(.5deg)}70%{transform:scale(1.04)}100%{opacity:1;transform:scale(1) rotate(0)}}
@keyframes hot-brand{0%{opacity:0;transform:scale(1.8) rotate(-10deg)}40%{transform:scale(.9) rotate(1.5deg)}65%{transform:scale(1.06)}100%{opacity:1;transform:scale(1) rotate(0)}}
@keyframes wool-breathe{0%,100%{transform:scale(1)}35%,65%{transform:scale(1.038)}}
@keyframes heart-lb-dub{0%,100%{transform:scale(1)}7%{transform:scale(1.16)}16%{transform:scale(1.01)}24%{transform:scale(1.09)}38%{transform:scale(1)}}
@keyframes wheat-shimmer{0%{background-position:-200% 0}100%{background-position:200% 0}}
@keyframes lantern-flicker{0%,100%{box-shadow:0 0 8px rgba(184,146,42,.2)}35%{box-shadow:0 0 18px rgba(184,146,42,.45)}65%{box-shadow:0 0 12px rgba(184,146,42,.32)}}
@keyframes wool-drift{0%,100%{transform:translateY(0) rotate(0deg)}30%{transform:translateY(-6px) rotate(-4deg)}60%{transform:translateY(-4px) rotate(3deg)}80%{transform:translateY(-7px) rotate(-1.5deg)}}
@keyframes scale-settle{0%,85%,100%{transform:translateY(0)}88%{transform:translateY(-2px)}94%{transform:translateY(1.2px)}}
@keyframes ct-sweep{0%{top:-2px;opacity:0}4%{opacity:.75}90%{opacity:.45}100%{top:100%;opacity:0}}
@keyframes sonar-pulse{0%{transform:scale(1);opacity:.9}100%{transform:scale(3.5);opacity:0}}
@keyframes sign-glint{0%{left:-100%}100%{left:160%}}
@keyframes lamp-breath{0%,100%{opacity:.25}45%{opacity:.95}55%{opacity:.8}}
@keyframes golden-drift{0%{left:-90%}100%{left:140%}}
@keyframes resting-pulse{0%,100%{box-shadow:0 0 0 1px var(--bd),0 2px 8px rgba(0,0,0,.3)}50%{box-shadow:0 0 0 1px rgba(184,146,42,.14),0 4px 18px rgba(184,146,42,.055)}}
@keyframes reading-flicker{0%,100%{box-shadow:0 0 0 1px var(--bd)}50%{box-shadow:0 0 0 1px rgba(184,146,42,.18),0 3px 14px rgba(184,146,42,.07)}}
@keyframes golden-pulse{0%,100%{box-shadow:0 2px 18px rgba(184,146,42,.45)}50%{box-shadow:0 5px 36px rgba(184,146,42,.75)}}
@keyframes warm-glow{0%,100%{box-shadow:0 2px 10px rgba(184,146,42,.3)}50%{box-shadow:0 4px 24px rgba(184,146,42,.55)}}
@keyframes ledger-entry{0%{opacity:0;transform:translateX(-8px)}100%{opacity:1;transform:translateX(0)}}
@keyframes weigh-in{0%{opacity:0;transform:translateY(10px) scale(.97)}100%{opacity:1;transform:translateY(0) scale(1)}}
@keyframes card-draw{0%{opacity:0;transform:translateY(12px) scale(.96)}60%{transform:translateY(-2px) scale(1.01)}100%{opacity:1;transform:translateY(0) scale(1)}}
@keyframes page-fan{0%{opacity:0;transform:translateY(16px) rotate(-.4deg)}65%{transform:translateY(-2px) rotate(.1deg)}100%{opacity:1;transform:translateY(0) rotate(0)}}
@keyframes scan-in{0%{opacity:0;clip-path:inset(0 0 100% 0)}100%{opacity:1;clip-path:inset(0 0 0% 0)}}
@keyframes fadeUp{0%{opacity:0;transform:translateY(6px)}100%{opacity:1;transform:translateY(0)}}
@keyframes approve{0%{opacity:0;transform:scale(1.5) rotate(-8deg)}55%{transform:scale(.96) rotate(.5deg)}100%{opacity:1;transform:scale(1) rotate(0)}}
.panel{animation:weigh-in .4s cubic-bezier(.4,0,.2,1) both,panel-breathe 7s ease-in-out infinite}
.panel:nth-child(2){animation-delay:.09s,1.5s}
.panel:nth-child(3){animation-delay:.18s,3s}
.view.active{animation:rise-from-earth .45s cubic-bezier(.25,0,.3,1) both}
.sidebar{position:relative;overflow:hidden}
.sidebar::before{content:'';position:absolute;right:0;top:0;width:1px;height:100%;background:linear-gradient(to bottom,transparent,rgba(184,146,42,.5) 30%,rgba(232,197,90,.7) 50%,rgba(184,146,42,.5) 70%,transparent);animation:sidebar-edge 4s ease-in-out infinite;z-index:2;pointer-events:none}
@keyframes sidebar-edge{0%,100%{opacity:.35}50%{opacity:1}}
.ni{transition:background .22s,color .22s,border-color .22s,transform .22s;position:relative;overflow:hidden}
.ni::before{content:'';position:absolute;left:0;top:0;bottom:0;width:0;background:linear-gradient(90deg,rgba(184,146,42,.35),transparent);border-radius:2px 0 0 2px;transition:width .2s}
.ni.active::before{width:3px;background:linear-gradient(to bottom,var(--orb),var(--orl),var(--or))}
.ni:hover{transform:translateX(4px);background:var(--s2);color:var(--tx)}
.ni.active{background:linear-gradient(90deg,rgba(184,146,42,.22),rgba(232,197,90,.08),rgba(184,146,42,.22))!important;background-size:200% 100%!important;animation:ledger-entry .3s ease both,ni-shimmer 4s ease-in-out infinite;border-color:rgba(184,146,42,.5)}
.ni::after{content:'';position:absolute;top:0;left:-100%;width:55%;height:100%;background:linear-gradient(90deg,transparent,rgba(184,146,42,.12),transparent);pointer-events:none}
.ni:hover::after{animation:sign-glint .45s ease}
@keyframes ni-shimmer{0%{background-position:-200% 0}100%{background-position:200% 0}}
.logo-icon{animation:lantern-flicker 4s ease-in-out infinite;transition:transform .22s,box-shadow .22s}
.logo-icon:hover{transform:scale(1.12);box-shadow:0 0 18px rgba(184,146,42,.55)}
.topbar{position:relative;overflow:hidden}
.topbar::before{content:'';position:absolute;bottom:0;left:-100%;width:50%;height:1px;background:linear-gradient(90deg,transparent,rgba(232,197,90,.6),transparent);animation:topbar-sweep 6s ease-in-out infinite}
.topbar::after{content:'';position:absolute;bottom:0;top:0;width:35%;background:linear-gradient(90deg,transparent,rgba(184,146,42,.045),transparent);animation:golden-drift 8s ease-in-out infinite;pointer-events:none}
@keyframes topbar-sweep{0%{left:-100%}100%{left:200%}}
@keyframes panel-breathe{0%,100%{box-shadow:0 0 0 1px var(--bd)}50%{box-shadow:0 0 0 1px rgba(184,146,42,.18),0 4px 24px rgba(184,146,42,.07)}}
.upload-zone{transition:background .22s,border-color .22s;position:relative;overflow:hidden}
.upload-zone::after{content:'';position:absolute;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent 0%,rgba(184,146,42,.6) 30%,rgba(232,197,90,.9) 50%,rgba(184,146,42,.6) 70%,transparent);animation:scan-line 3.5s ease-in-out infinite;pointer-events:none;z-index:1}
@keyframes scan-line{0%{top:-4px;opacity:0}5%{opacity:1}90%{opacity:.6}100%{top:calc(100% + 4px);opacity:0}}
.uicon{transition:transform .25s,box-shadow .25s;animation:wool-breathe 4.5s ease-in-out infinite}
.upload-zone:hover .uicon{transform:scale(1.1);box-shadow:0 0 18px rgba(184,146,42,.45)}
.btn{transition:background .15s,box-shadow .2s,transform .13s}
.btn:not(:disabled):hover{box-shadow:0 4px 18px rgba(184,146,42,.5);transform:translateY(-1px)}
.btn:not(:disabled):active{transform:translateY(1px);box-shadow:none}
.btn2{transition:border-color .2s,color .2s,transform .13s}
.btn2:hover{border-color:rgba(122,106,69,.65);color:var(--tx);transform:translateY(-1px)}
@keyframes ha-halo{0%,100%{box-shadow:0 2px 22px rgba(184,146,42,.5),0 0 0 1px rgba(200,165,48,.3)}50%{box-shadow:0 6px 40px rgba(184,146,42,.8),0 0 0 2px rgba(200,165,48,.55)}}
.ha-btn{animation:ha-halo 3.5s ease-in-out infinite;transition:transform .16s,background .2s}
.ha-btn:hover{transform:translateY(-2px)}
.ha-btn:active{transform:translateY(1px)}
@keyframes btn-halo{0%,100%{box-shadow:0 2px 10px rgba(184,146,42,.3)}50%{box-shadow:0 4px 28px rgba(184,146,42,.6),0 0 0 1px rgba(200,165,48,.25)}}
.analyze-btn{transition:background .2s,box-shadow .22s,transform .13s}
.analyze-btn:not(:disabled){animation:btn-halo 3s ease-in-out infinite}
.analyze-btn:not(:disabled):hover{box-shadow:0 6px 28px rgba(184,146,42,.6);transform:translateY(-1px)}
.sb{transition:border-color .22s,background .22s,box-shadow .25s,transform .2s}
.sb:hover{transform:translateY(-3px);border-color:rgba(184,146,42,.6);background:#141109;box-shadow:0 6px 16px rgba(0,0,0,.55)}
.sb.on .sb-ico{animation:wool-drift 3.2s ease-in-out infinite}
@keyframes symptom-pulse{0%,100%{box-shadow:0 0 12px rgba(184,146,42,.25),inset 0 0 8px rgba(184,146,42,.08)}50%{box-shadow:0 0 24px rgba(184,146,42,.5),inset 0 0 16px rgba(184,146,42,.18)}}
@keyframes sb-confirm{0%{transform:scale(1)}35%{transform:scale(.93)}65%{transform:scale(1.05)}100%{transform:scale(1)}}
.sb.on{animation:sb-confirm .38s cubic-bezier(.4,0,.2,1) both,symptom-pulse 2.6s ease-in-out infinite}
.sb-ico{transition:transform .22s,filter .22s,color .22s}
.sb:hover .sb-ico{transform:scale(1.16)}
.sb-ck{animation:hot-brand .38s cubic-bezier(.4,0,.2,1) both}
.dc{transition:border-color .22s,box-shadow .22s,transform .2s}
.dc:hover{transform:translateY(-3px);border-color:rgba(184,146,42,.42);box-shadow:0 10px 28px rgba(0,0,0,.6)}
.dc-icon svg{transition:transform .22s}
.dc:hover .dc-icon svg{transform:scale(1.14) rotate(-5deg)}
.dc-tb{transition:background .16s,border-color .16s,color .16s,transform .13s}
.dc-tb:hover{transform:translateY(-1px);background:var(--ord)}
@keyframes readout-flicker{0%,95%,100%{opacity:1}96%{opacity:.6}97%{opacity:1}98.5%{opacity:.8}}
.gn{animation:readout-flicker 6s ease-in-out infinite}
.dc-pct{animation:readout-flicker 9s ease-in-out infinite}
.cbf{animation:trough-fill 1s cubic-bezier(.4,0,.2,1) both;position:relative;overflow:hidden}
@keyframes trough-fill{0%{width:0!important}}
@keyframes bar-shimmer{0%{background-position:-200% 0}100%{background-position:200% 0}}
.cbf::after{content:'';position:absolute;inset:0;background:linear-gradient(90deg,transparent 20%,rgba(255,255,255,.4) 50%,transparent 80%);background-size:200%;animation:bar-shimmer 2.5s ease infinite .5s}
.hbadge{animation:hot-brand .42s cubic-bezier(.4,0,.2,1) .18s both}
@keyframes heartbeat{0%,100%{transform:scale(1)}14%{transform:scale(1.14)}28%{transform:scale(1)}42%{transform:scale(1.07)}70%{transform:scale(1)}}
.hbadge.healthy{animation:approve .4s cubic-bezier(.4,0,.2,1) .15s both,heartbeat 2.4s ease-in-out 1.2s infinite}
.rov.show-anim{animation:earmark .42s cubic-bezier(.4,0,.2,1) both}
.fi{animation:brand-burn .32s ease both;transition:color .16s,transform .16s}
.fi:nth-child(1){animation-delay:.05s}.fi:nth-child(2){animation-delay:.10s}.fi:nth-child(3){animation-delay:.15s}.fi:nth-child(4){animation-delay:.20s}.fi:nth-child(5){animation-delay:.25s}.fi:nth-child(6){animation-delay:.30s}
.fi:hover{transform:translateX(5px)}
.fd{transition:transform .22s,border-color .22s}
.fi:hover .fd{transform:scale(1.28) rotate(90deg);border-color:var(--orl)}
.rc{transition:border-color .2s,box-shadow .2s,transform .18s}
.rc:hover{border-color:rgba(184,146,42,.52);transform:translateY(-2px);box-shadow:0 7px 20px rgba(0,0,0,.55)}
.rlc{transition:border-color .2s,box-shadow .2s,transform .18s}
.rlc:hover{border-color:rgba(184,146,42,.48);transform:translateY(-4px);box-shadow:0 12px 28px rgba(0,0,0,.6)}
.rlci{overflow:hidden}
.rlci img{transition:transform .45s ease}
.rlc:hover .rlci img{transform:scale(1.08)}
.del-btn,.rlc-del{transition:opacity .2s,transform .2s}
.del-btn:hover,.rlc-del:hover{transform:scale(1.22) rotate(12deg)!important}
.mft{transition:background .2s,border-color .2s,color .2s,transform .15s}
.mft:hover{transform:translateY(-1px)}
.mft.active{animation:hot-brand .3s ease both}
.mcard{animation:instrument-glow 5s ease-in-out infinite;transition:transform .2s,box-shadow .2s;position:relative}
@keyframes instrument-glow{0%,100%{box-shadow:0 0 0 1px var(--bd)}50%{box-shadow:0 0 0 1px rgba(184,146,42,.2),0 4px 20px rgba(184,146,42,.08)}}
.mcard::before{content:'';position:absolute;top:0;left:8%;right:8%;height:1px;background:linear-gradient(90deg,transparent,rgba(184,146,42,.3),transparent)}
.mcard:hover{transform:translateY(-2px);box-shadow:0 9px 26px rgba(0,0,0,.55)}
.ifield,.fi2{transition:border-color .2s,box-shadow .2s}
.ifield:focus,.fi2:focus{border-color:var(--or);box-shadow:0 0 0 2.5px rgba(184,146,42,.18)}
.fsel:focus{border-color:var(--or);box-shadow:0 0 0 2.5px rgba(184,146,42,.18)}
.modal-ov.open .modal{animation:awning-drop .4s cubic-bezier(.4,0,.2,1) both}
@keyframes awning-drop{0%{opacity:0;transform:translateY(-20px) scaleY(.88);transform-origin:top center}65%{transform:translateY(4px) scaleY(1.015)}100%{opacity:1;transform:translateY(0) scaleY(1)}}
.reason-box{animation:brand-burn .32s ease .1s both;border-left:2px solid var(--or)}
.mprice{display:inline-block;animation:weight-stamp .48s cubic-bezier(.4,0,.2,1) both}
.diet-item{transition:background .15s,transform .15s;border-radius:4px;padding-left:4px}
.diet-item:hover{background:rgba(184,146,42,.07);transform:translateX(5px)}
.d-dot{transition:transform .22s,box-shadow .22s}
.diet-item:hover .d-dot{transform:scale(1.4);box-shadow:0 0 12px rgba(212,170,63,.8)}
.btn-treat{transition:border-color .2s,color .2s,box-shadow .2s,transform .13s}
.btn-treat:hover{border-color:var(--gr);box-shadow:0 0 14px rgba(61,214,140,.22);transform:translateY(-1px)}
.btn-vet{transition:background .2s,box-shadow .2s,transform .13s}
.btn-vet:hover{box-shadow:0 7px 22px rgba(184,146,42,.58);transform:translateY(-1px)}
.live-dot{position:relative}
.live-dot::after{content:'';position:absolute;inset:-3px;border-radius:50%;background:var(--gr);opacity:0;animation:sonar-pulse 2.4s ease-out infinite}
.center::-webkit-scrollbar{width:4px}
.center::-webkit-scrollbar-thumb{background:var(--bd);border-radius:2px}
.center:hover::-webkit-scrollbar-thumb{background:rgba(184,146,42,.28)}
body::after{content:'';position:fixed;inset:0;pointer-events:none;z-index:3;background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");background-repeat:repeat;background-size:160px;opacity:.028;mix-blend-mode:overlay}
@keyframes gold-sweep{0%{background-position:-200% 0}100%{background-position:200% 0}}
@keyframes ember{0%,100%{box-shadow:0 0 0 0 rgba(184,146,42,0)}50%{box-shadow:0 0 0 5px rgba(184,146,42,0),0 0 12px rgba(184,146,42,.35)}}
.logo-icon{animation:ember 3.5s ease-in-out infinite}
.price-layout{display:flex;gap:24px;align-items:flex-start}
.price-form{width:340px;min-width:280px;flex-shrink:0}
.price-formula{background:var(--s3);border:1px solid var(--bd);border-radius:7px;padding:10px 13px;margin-bottom:16px;display:flex;flex-direction:column;gap:4px;font-size:11px;color:var(--tm);line-height:1.6}
.price-calc-btn{width:100%;margin-bottom:12px;padding:11px 16px;font-size:15px}
.price-result-col{flex:1;min-width:0}
.price-result-header{background:var(--s2);border:1px solid var(--bd);border-radius:8px;padding:16px;margin-bottom:12px;text-align:center}
.price-est{font-size:11px;color:var(--tm);margin-top:4px}
.price-placeholder{padding:28px 20px;text-align:center;color:var(--td);font-size:11px;font-family:var(--font-body);border:1px dashed var(--bd);border-radius:7px}
.btn,.btn2,.btn-danger{height:34px;display:inline-flex;align-items:center;justify-content:center;white-space:nowrap}
.upload-zone,.result-area{min-height:200px}
.rlg{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.mtable{table-layout:fixed;word-break:break-word}
.mcard{padding:14px}
.breakdown{padding:10px 12px}
.sidebar{flex-shrink:0}
.content{min-height:0}
.panel,.mcard,.rc,.rlc,.dc,.mc{border-radius:8px}
.panel{border-radius:10px}
.ifield,.fi2,.fsel{height:36px;padding:0 10px;box-sizing:border-box}
textarea.ifield,textarea.obs-i{height:auto!important;padding:8px 10px}
.sec-t{margin-bottom:12px}
.ph{margin-bottom:12px}
.sym-grid,.sym-grid2{overflow:hidden}
.mtable th{height:34px;vertical-align:middle}
.mtable td{height:38px;vertical-align:middle}
.mft{height:28px;display:inline-flex;align-items:center}
.eid-banner{align-items:center}
.rc{min-height:60px}
.del-btn{top:50%;transform:translateY(-50%);right:6px}
.rc:hover .del-btn{transform:translateY(-50%) scale(1.15)}
.modal{width:420px}
.feat-grid{gap:6px;margin-top:10px}
.reason-box{margin-top:10px;padding:10px 12px}
input.ifield,input.fi2{line-height:36px}
select.ifield,select.fi2,.fsel{appearance:none;line-height:normal;cursor:pointer}
select.ifield option,select.fi2 option,.fsel option{background:var(--s2);color:var(--tx)}
.view.active{gap:14px}
.panel+.panel{margin-top:0}
.price-result-col{min-height:200px}
.price-result-header{text-align:center;padding:20px 16px}
.recs-grid,.rlg{min-height:60px}
.mtable-wrap{overflow-x:auto;-webkit-overflow-scrolling:touch}
#view-scan .grid2{grid-template-columns:1fr 1fr;align-items:start}
#view-health .panel{width:100%}
.topbar h1{margin:0;line-height:48px}
.center::-webkit-scrollbar{width:4px}
.rsidebar::-webkit-scrollbar{width:3px}
.rsidebar::-webkit-scrollbar-thumb{background:var(--bd);border-radius:2px}
@media(max-width:900px){.grid2{grid-template-columns:1fr}}
@media(max-width:1100px){.rlg{grid-template-columns:repeat(2,1fr)}}
@media(max-width:700px){.rlg{grid-template-columns:1fr}.price-layout{flex-direction:column}.price-form{width:100%}}
</style>
</head>
<body>
<canvas id="bg-canvas" style="position:fixed;inset:0;width:100%;height:100%;pointer-events:none;z-index:0"></canvas>
<aside class="sidebar" style="position:relative;z-index:1">
  <div class="logo">
  <div class="logo-icon" id="logo-mark">
    <svg width="36" height="36" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
      <g id="lgo-hex"><polygon points="18,2 31,9.5 31,26.5 18,34 5,26.5 5,9.5" fill="none" stroke="rgba(184,146,42,0.35)" stroke-width="0.7"/><polygon points="18,4 29.5,10.5 29.5,25.5 18,32 6.5,25.5 6.5,10.5" fill="none" stroke="rgba(232,197,90,0.18)" stroke-width="0.4"/></g>
      <g stroke="rgba(184,146,42,0.6)" stroke-width="0.8" stroke-linecap="round"><line x1="18" y1="2" x2="18" y2="5.5"/><line x1="31" y1="9.5" x2="28" y2="11.2"/><line x1="31" y1="26.5" x2="28" y2="24.8"/><line x1="18" y1="34" x2="18" y2="30.5"/><line x1="5" y1="26.5" x2="8" y2="24.8"/><line x1="5" y1="9.5" x2="8" y2="11.2"/></g>
      <circle cx="18" cy="18" r="13.5" fill="none" stroke="rgba(184,146,42,0.22)" stroke-width="0.6" stroke-dasharray="3 5"/>
      <rect x="8.5" y="14" width="15" height="10" rx="5" fill="rgba(232,197,90,0.12)" stroke="rgba(212,170,63,0.7)" stroke-width="0.9"/>
      <circle cx="11" cy="14.5" r="2.4" fill="rgba(184,146,42,0.15)" stroke="rgba(212,170,63,0.5)" stroke-width="0.7"/>
      <circle cx="14.5" cy="13.5" r="2.7" fill="rgba(184,146,42,0.18)" stroke="rgba(212,170,63,0.55)" stroke-width="0.7"/>
      <circle cx="18.5" cy="13.2" r="2.5" fill="rgba(184,146,42,0.16)" stroke="rgba(212,170,63,0.5)" stroke-width="0.7"/>
      <circle cx="22" cy="14" r="2.1" fill="rgba(184,146,42,0.13)" stroke="rgba(212,170,63,0.45)" stroke-width="0.7"/>
      <ellipse cx="25.2" cy="17.2" rx="2.6" ry="2.2" fill="rgba(200,158,44,0.2)" stroke="rgba(212,170,63,0.75)" stroke-width="0.85"/>
      <path d="M24.2 15.4 L25.6 13.4 L26.6 15.2 Z" fill="rgba(184,146,42,0.25)" stroke="rgba(212,170,63,0.6)" stroke-width="0.7"/>
      <circle cx="26.1" cy="17" r="0.9" fill="rgba(232,197,90,0.9)"/>
      <circle cx="26.3" cy="17" r="0.35" fill="rgba(10,8,4,0.8)"/>
      <g stroke="rgba(184,146,42,0.65)" stroke-width="0.85" stroke-linecap="round"><line x1="11.5" y1="23.5" x2="10.8" y2="27.8"/><line x1="14" y1="24" x2="13.5" y2="28"/><line x1="17.5" y1="24" x2="17.2" y2="28"/><line x1="20.5" y1="23.5" x2="20.8" y2="27.8"/></g>
      <path d="M8.8 17.5 Q6.8 16.2 7.2 14.8 Q7.6 13.4 9.2 14.2" fill="none" stroke="rgba(184,146,42,0.55)" stroke-width="0.85" stroke-linecap="round"/>
      <g stroke="rgba(184,146,42,0.45)" stroke-width="0.7" stroke-linecap="round" fill="none"><path d="M5.5 12 L5.5 8 L9.5 8"/><path d="M30.5 12 L30.5 8 L26.5 8"/><path d="M5.5 24 L5.5 28 L9.5 28"/><path d="M30.5 24 L30.5 28 L26.5 28"/></g>
    </svg>
  </div>
  <div class="logo-text-block"><span class="logo-main">SHEPPY</span></div>
</div>
  <nav class="nav">
    <div class="ni active" onclick="sv('scan',this)" data-key="nav_scan">Scan Sheep</div>
    <div class="ni" onclick="sv('health',this)" data-key="nav_health">Health Advisor</div>
    <div class="ni" onclick="sv('records',this)" data-key="nav_records">Records</div>
    <div class="ni" onclick="sv('market',this)" data-key="nav_market">Market Price</div>
    <div class="ni" onclick="sv('price',this)" data-key="nav_price">Price Calculator</div>
  </nav>
  <div class="lang-box">
    <div class="lt" data-key="lang_label">Language</div>
    <div class="lo active" onclick="setLang(this,'en')">English</div>
    <div class="lo" onclick="setLang(this,'ur')">اردو</div>
    <div class="lo" onclick="setLang(this,'sd')">سنڌي</div>
  </div>
</aside>
<div class="main" style="position:relative;z-index:1">
  <div class="topbar"><h1 id="page-title">Dashboard</h1></div>
  <div class="content">
    <div class="center">
      <!-- SCAN -->
      <div class="view active" id="view-scan">
        <div class="panel">
          <div class="ph"><span class="pt" data-key="scan_panel">Sheep Analysis Panel</span></div>
          <div class="grid2">
            <div class="upload-zone" id="uzone" ondragover="event.preventDefault();this.classList.add('drag')" ondragleave="this.classList.remove('drag')" ondrop="onDrop(event)">
              <input type="file" id="finput" accept="image/*" onchange="onFile(event)">
              <img id="uprev" class="prev-img" alt=""><span class="prev-badge" id="pbadge" data-key="img_loaded">Image loaded</span>
              <div id="img-actions" style="display:none;position:absolute;top:8px;right:8px;gap:5px;z-index:10;flex-direction:row;align-items:center;background:rgba(0,0,0,.55);border-radius:6px;padding:3px;backdrop-filter:blur(4px)">
                <button onclick="event.stopPropagation();finput.click()" style="background:rgba(184,120,12,.85);border:none;color:#fff;font-size:10px;font-weight:600;padding:4px 9px;border-radius:4px;cursor:pointer;white-space:nowrap" data-key="btn_change">Change</button>
                <button onclick="event.stopPropagation();clearImage()" style="background:rgba(224,85,85,.85);border:none;color:#fff;font-size:10px;font-weight:600;padding:4px 9px;border-radius:4px;cursor:pointer;white-space:nowrap" data-key="btn_remove">Remove</button>
              </div>
              <div id="uph"><div class="uicon">🐑</div><p data-key="upload_prompt">Upload or Drag Sheep Photo</p><small>JPEG · PNG · WebP · Max 15MB</small><button class="btn" style="margin-top:7px" onclick="event.stopPropagation();finput.click()" data-key="btn_browse">Browse</button></div>
            </div>
            <div class="result-area">
              <div class="rph" id="rph">🐑</div><img id="rimg" class="rimg" alt="">
              <div class="rov" id="rov">
                <div class="or2"><span class="ol" data-key="lbl_breed">Breed</span><span class="ov" id="ov-breed" style="color:var(--or)">—</span></div>
                <div class="or2"><span class="ol" data-key="lbl_use">Use</span><span class="ov" id="ov-use">—</span></div>
                <div class="or2"><span class="ol" data-key="lbl_conf">Confidence</span><span class="ov" id="ov-conf">—</span></div>
                <span id="ov-hbadge" class="hbadge healthy">—</span>
                <div id="ov-bars" style="margin-top:5px"></div>
              </div>
              <button class="analyze-btn" id="abtn" onclick="runAnalysis()" disabled data-key="btn_analyze">Analyze Sheep</button>
            </div>
          </div>
          <div class="feat-grid" id="feat-grid" style="display:none">
            <div class="fi"><span class="fd">+</span><span data-key="lbl_breed">Breed</span>: <strong style="color:var(--or);margin-left:3px" id="f-breed">—</strong></div>
            <div class="fi"><span class="fd">+</span><span data-key="lbl_use">Use</span>: <strong style="margin-left:3px" id="f-use2">—</strong></div>
            <div class="fi"><span class="fd">+</span><span data-key="lbl_health">Health</span>: <strong style="margin-left:3px" id="f-hlth">—</strong></div>
            <div class="fi"><span class="fd">+</span><span data-key="lbl_wool">Wool</span>: <strong style="margin-left:3px" id="f-wool">—</strong></div>
            <div class="fi"><span class="fd">+</span>BCS: <strong style="margin-left:3px" id="f-bcs">—</strong></div>
            <div class="fi"><span class="fd">+</span>Pipeline: <strong style="margin-left:3px;font-size:9px;color:var(--tm)" id="f-pipe">—</strong></div>
          </div>
          <div class="reason-box" id="reason-box" style="display:none"><strong data-key="why_breed">Why this breed?</strong> <span id="reason-text"></span></div>
        </div>
        <div class="panel">
          <div class="ph"><span class="pt" data-key="recent_records">Recent Records</span><div style="display:flex;gap:7px"><button class="btn" onclick="openModal()" data-key="btn_add">+ Add</button><button class="btn-danger" id="clear-btn" onclick="clearAllRecs()" style="display:none" data-key="btn_clear_all">Clear All</button></div></div>
          <div class="recs-grid" id="recs-mini"></div>
        </div>
      </div>
      <!-- HEALTH -->
      <div class="view" id="view-health">
        <div class="panel ha-panel">
          <div class="ha-title" data-key="ha_title">Sheep Health Advisor</div>
          <div class="sym-grid" id="sr1"></div>
          <div class="sym-grid2" id="sr2"></div>
          <input class="obs-i" id="obs-i" data-placeholder-key="obs_placeholder" placeholder="Other Observations (optional)…">
          <button class="ha-btn" id="ha-btn" onclick="runDiag()" data-key="btn_analyze_health">Analyze Health</button>
          <div id="ha-res" style="display:none">
            <div class="ha-results-wrap">
              <div class="ha-results-left">
                <div class="sec-t" data-key="disease_section">Detailed Disease Predictions (Probability-Sorted)</div>
                <div id="dcl"></div>
                <div class="mini-grid" id="dcm"></div>
              </div>
              <div class="ha-results-right">
                <div class="ha-gauge-block">
                  <div class="gauge-wrap">
                    <svg viewBox="0 0 148 100" xmlns="http://www.w3.org/2000/svg" overflow="visible">
                      <defs><filter id="gf" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="3" result="blur"/><feComposite in="SourceGraphic" in2="blur" operator="over"/></filter></defs>
                      <text x="2" y="90" fill="#3a3320" font-size="8" font-family="IBM Plex Mono,monospace">0</text>
                      <text x="18" y="44" fill="#3a3320" font-size="8" font-family="IBM Plex Mono,monospace">20</text>
                      <text x="60" y="18" fill="#3a3320" font-size="8" font-family="IBM Plex Mono,monospace">50</text>
                      <text x="98" y="44" fill="#3a3320" font-size="8" font-family="IBM Plex Mono,monospace">80</text>
                      <text x="118" y="90" fill="#3a3320" font-size="8" font-family="IBM Plex Mono,monospace">100</text>
                      <path d="M 14 85 A 60 60 0 0 1 134 85" fill="none" stroke="#1e1b10" stroke-width="12" stroke-linecap="round"/>
                      <path id="garc-glow" d="M 14 85 A 60 60 0 0 1 134 85" fill="none" stroke="#d4aa3f" stroke-width="14" stroke-linecap="round" stroke-dasharray="0 188.5" opacity="0.35" style="filter:blur(4px);transition:stroke-dasharray 1s ease,stroke .5s ease"/>
                      <path id="garc" d="M 14 85 A 60 60 0 0 1 134 85" fill="none" stroke="#d4aa3f" stroke-width="12" stroke-linecap="round" stroke-dasharray="0 188.5" style="transition:stroke-dasharray 1s ease,stroke .5s ease"/>
                      <line x1="22" y1="72" x2="27" y2="65" stroke="#2a2616" stroke-width="1.5"/><line x1="42" y1="38" x2="46" y2="43" stroke="#2a2616" stroke-width="1.5"/><line x1="106" y1="38" x2="102" y2="43" stroke="#2a2616" stroke-width="1.5"/><line x1="126" y1="72" x2="121" y2="65" stroke="#2a2616" stroke-width="1.5"/>
                    </svg>
                    <div class="gauge-center"><div class="gn" id="gnum">—</div><div class="gs" data-key="score_lbl">SCORE</div></div>
                  </div>
                  <div class="g-status" id="gstatus">—</div>
                </div>
                <div class="hs-panel" id="chart-panel" style="display:none"><div class="sec-t" style="margin-bottom:5px" data-key="prob_chart">Disease Probability Chart</div><svg class="spark-svg" id="spark" viewBox="0 0 200 68" xmlns="http://www.w3.org/2000/svg" overflow="visible"></svg></div>
                <div class="hs-panel" id="diet-panel" style="display:none"><div class="sec-t" style="margin-bottom:6px" data-key="diet_plan">Suggested Diet Plan</div><div id="diet"></div></div>
                <div class="consult-wrap" id="consult"><div class="ct" id="ct"></div><div id="cb"></div><div class="c-btns"><button class="btn-treat" onclick="markTx(this)" data-key="btn_treated">Mark as Treated</button><button class="btn-vet" data-key="btn_vet">Consult Vet</button></div></div>
                <div class="disclaim" data-key="disclaimer">Medical Disclaimer: This is a symptom based prediction. Accuracy is limited, always consult a veterinarian for confirmation.</div>
              </div>
            </div>
          </div>
        </div>
      </div>
      <!-- RECORDS -->
      <div class="view" id="view-records">
        <div class="panel">
          <div class="ph"><span class="pt" data-key="all_records">All Records <span style="color:var(--tm);font-weight:400;font-size:11px" id="rec-count"></span></span><div style="display:flex;gap:7px"><button class="btn" onclick="openModal()" data-key="btn_add">+ Add</button><button class="btn2" onclick="exportCSV()" data-key="btn_export">Export CSV</button><button class="btn-danger" onclick="clearAllRecs()" data-key="btn_clear_all">Clear All</button></div></div>
          <div class="rlg" id="recs-large"></div>
        </div>
      </div>
      <!-- MARKET -->
      <div class="view" id="view-market">
        <div class="panel">
          <div class="ph"><span class="pt"><span class="live-dot"></span><span data-key="mkt_title">Mandi Rate Table (PKR/kg)</span> <span id="mkt-upd" style="font-size:10px;color:var(--tm);font-weight:400"></span></span><button class="btn2" onclick="loadMarket(true)" data-key="btn_refresh">↻ Refresh</button></div>
          <div class="eid-banner" id="eid-banner" data-key="eid_msg">🌙 Eid ul Adha premium active — all rates ×1.6</div>
          <div class="mkt-filter">
            <span class="mft active" onclick="mktF(this,'adult')" data-key="age_adult">Adult</span>
            <span class="mft" onclick="mktF(this,'young')" data-key="age_young">Young</span>
            <span class="mft" onclick="mktF(this,'lamb')" data-key="age_lamb">Lamb</span>
            <span class="mft" onclick="mktF(this,'old')" data-key="age_old">Old</span>
          </div>
          <div id="mkt-load" style="text-align:center;padding:28px;color:var(--tm);font-size:12px"><span class="spin"></span> Loading…</div>
          <div class="mtable-wrap"><table class="mtable" id="mkt-tbl" style="display:none"><thead><tr id="mkt-th"></tr></thead><tbody id="mtbody"></tbody></table></div>
        </div>
      </div>
      <!-- PRICE CALCULATOR -->
      <div class="view" id="view-price">
        <div class="panel">
          <div class="ph"><span class="pt" data-key="price_calc">Price Calculator</span></div>
          <div class="price-layout">
            <div class="price-form">
              <div class="price-formula"><span data-key="mandi_formula">Mandi formula:</span><span style="color:var(--or);font-family:var(--font-mono);font-size:12px">Price = Weight × Rate/kg</span></div>
              <div class="fg"><label class="fl" data-key="weight_lbl">Live Weight (kg) *</label><input type="number" class="ifield" id="w-in" placeholder="e.g. 45" min="1" max="300" oninput="clrPR()"></div>
              <div class="fg"><label class="fl" data-key="lbl_breed">Breed</label><select class="ifield" id="br-ov" onchange="clrPR()"><option value="">Select</option><option>Lohi</option><option>Kajli</option><option>Lari</option></select></div>
              <div class="fg"><label class="fl" data-key="lbl_health">Health</label><select class="ifield" id="hl-ov" onchange="clrPR()"><option value="">Select</option><option value="healthy" data-key="health_healthy">Healthy</option><option value="weak" data-key="health_weak">Weak</option><option value="diseased" data-key="health_diseased">Diseased</option></select></div>
              <button class="btn price-calc-btn" onclick="calcPrice()" data-key="btn_calc">Calculate Price</button>
              <div class="price-placeholder" id="pph" data-key="price_hint">Enter weight above and click Calculate</div>
            </div>
            <div class="price-result-col" id="pres" style="display:none">
              <div class="price-result-header"><div class="mprice" id="mvp"></div><div class="price-est" id="mve"></div><div class="eid-badge" id="eidb">🌙 Eid ul Adha — ×1.6 premium applied</div></div>
              <div class="breakdown">
                <div class="bd-title" data-key="breakdown">Breakdown</div>
                <div class="mrow" style="padding:5px 0"><span class="mkey" data-key="weight_lbl">Weight</span><span class="mval or" id="pr-w">—</span></div>
                <div class="mrow" style="padding:5px 0"><span class="mkey" data-key="base_rate">Base Rate/kg</span><span class="mval" id="pr-b">—</span></div>
                <div class="mrow" style="padding:5px 0"><span class="mkey" data-key="age_factor">Age Factor</span><span class="mval" id="pr-a">—</span></div>
                <div class="mrow" style="padding:5px 0"><span class="mkey" data-key="health_factor">Health Factor</span><span class="mval" id="pr-h">—</span></div>
                <div class="mrow" style="padding:5px 0"><span class="mkey" data-key="season">Season</span><span class="mval" id="pr-s">—</span></div>
                <div class="mrow" style="padding:5px 0;border:none"><span class="mkey" data-key="final_rate">Final Rate/kg</span><span class="mval or" id="pr-r">—</span></div>
                <div class="formula-tag" id="pr-f"></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    <!-- RIGHT SIDEBAR -->
    <aside class="rsidebar" id="rsidebar">
      <div class="mcard">
        <div class="mcard-title">🐑 <span data-key="ai_analysis">AI Analysis</span></div>
        <div class="mrow"><span class="mkey" data-key="lbl_breed">Breed</span><span class="mval or" id="mv-breed">—</span></div>
        <div class="mrow"><span class="mkey" data-key="lbl_use">Use</span><span class="mval" id="mv-use">—</span></div>
        <div class="mrow"><span class="mkey" data-key="lbl_health">Health</span><span class="mval gr" id="mv-hlth">—</span></div>
        <div class="mrow"><span class="mkey" data-key="lbl_conf">Confidence</span><span class="mval" id="mv-conf">—</span></div>
      </div>
    </aside>
  </div>
</div>

<!-- MODAL -->
<div class="modal-ov" style="z-index:1000" id="modal" onclick="if(event.target===this)closeModal()">
  <div class="modal">
    <div class="mhdr"><span class="pt" data-key="add_record">Add Sheep Record</span><div class="mcls" onclick="closeModal()">✕</div></div>
    <div class="mbdy">
      <div style="display:flex;flex-direction:column;gap:4px"><label class="fl" data-key="photo_lbl">Photo</label><div class="iub" onclick="document.getElementById('mimg-in').click()"><input type="file" id="mimg-in" accept="image/*" onchange="prevMI(event)"><img id="mimg-prev" class="ipt" alt=""><div id="mimg-ph" style="font-size:11px;color:var(--td)" data-key="click_upload">Click to upload</div></div></div>
      <div class="frow"><div style="display:flex;flex-direction:column;gap:4px"><label class="fl" data-key="name_tag">Name / Tag</label><input class="fi2" id="r-name" placeholder="Sheep #12"></div><div style="display:flex;flex-direction:column;gap:4px"><label class="fl" data-key="lbl_breed">Breed *</label><select class="fsel" id="r-breed"><option value="">Select…</option><option>Lohi</option><option>Kajli</option><option>Lari</option><option>Other</option></select></div></div>
      <div class="frow"><div style="display:flex;flex-direction:column;gap:4px"><label class="fl" data-key="weight_lbl">Weight (kg)</label><input class="fi2" id="r-weight" type="number" placeholder="45"></div><div style="display:flex;flex-direction:column;gap:4px"><label class="fl" data-key="lbl_use">Use</label><select class="fsel" id="r-use"><option>Meat</option><option>Wool</option><option>Breeding</option><option>Milk</option></select></div></div>
      <div class="frow"><div style="display:flex;flex-direction:column;gap:4px"><label class="fl" data-key="price_min">Price Min (PKR)</label><input class="fi2" id="r-pmin" type="number" placeholder="80000"></div><div style="display:flex;flex-direction:column;gap:4px"><label class="fl" data-key="price_max">Price Max (PKR)</label><input class="fi2" id="r-pmax" type="number" placeholder="150000"></div></div>
      <div class="frow"><div style="display:flex;flex-direction:column;gap:4px"><label class="fl" data-key="lbl_health">Health</label><select class="fsel" id="r-hlth"><option value="good">Healthy</option><option value="bad">Weak</option><option value="bad">Diseased</option></select></div><div style="display:flex;flex-direction:column;gap:4px"><label class="fl">Notes</label><textarea class="fi2" id="r-notes" rows="2" placeholder="Observations…"></textarea></div></div>
      <div id="r-err" style="font-size:11px;color:var(--re);display:none" data-key="err_breed">Please select a breed.</div>
    </div>
    <div class="mftr"><button class="btn2" onclick="closeModal()" data-key="btn_cancel">Cancel</button><button class="btn" onclick="submitRec()" data-key="btn_save">Save</button></div>
  </div>
</div>

<!-- BOTTOM NAV -->
<nav class="bottom-nav" id="bottom-nav">
  <button class="bn-item active" data-view="scan" onclick="sv('scan',this)">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="3"/><path d="M3 9h18M9 21V9"/></svg>
    <span data-key="nav_scan">Scan</span>
  </button>
  <button class="bn-item" data-view="health" onclick="sv('health',this)">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
    <span data-key="nav_health">Health</span>
  </button>
  <button class="bn-item" data-view="records" onclick="sv('records',this)">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="9" y1="13" x2="15" y2="13"/><line x1="9" y1="17" x2="13" y2="17"/></svg>
    <span data-key="nav_records">Records</span>
  </button>
  <button class="bn-item" data-view="market" onclick="sv('market',this)">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
    <span data-key="nav_market">Market</span>
  </button>
  <button class="bn-item" data-view="price" onclick="sv('price',this)">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 000 7h5a3.5 3.5 0 010 7H6"/></svg>
    <span data-key="nav_price">Price</span>
  </button>
</nav>

<script>
(function(){try{const s=JSON.parse(localStorage.getItem('sheep_recs')||'[]');if(s.some(r=>r.img&&!r.img.startsWith('data:')))localStorage.removeItem('sheep_recs');}catch{localStorage.removeItem('sheep_recs');}})();
if(!AbortSignal.timeout){AbortSignal.timeout=function(ms){var c=new AbortController();setTimeout(()=>c.abort(),ms);return c.signal;};}
const $=id=>document.getElementById(id);
const show=(id,d='block')=>{const e=$(id);if(e)e.style.display=d};
const hide=id=>{const e=$(id);if(e)e.style.display='none'};
const cap=s=>s?s[0].toUpperCase()+s.slice(1):'—';
let recs=lrecs(),curFile=null,modalImg=null,mktData=null,mktAge='adult';
window._AI={};
const SC={Critical:'#e05555',High:'#e8a020',Moderate:'#3dd68c'};
const SBG={Critical:'rgba(224,85,85,.14)',High:'rgba(232,160,32,.12)',Moderate:'rgba(61,214,140,.1)'};
const SBD={Critical:'rgba(224,85,85,.4)',High:'rgba(232,160,32,.3)',Moderate:'rgba(61,214,140,.28)'};
const LANGS={
  en:{nav_scan:"Scan Sheep",nav_health:"Health Advisor",nav_records:"Records",nav_market:"Market Price",nav_price:"Price Calculator",lang_label:"Language",scan_panel:"Sheep Analysis Panel",btn_add:"Add Record",img_loaded:"Image loaded",btn_change:"Change",btn_remove:"Remove",upload_prompt:"Upload or Drag Sheep Photo",btn_browse:"Browse",lbl_breed:"Breed",lbl_conf:"Confidence",lbl_health:"Health",lbl_wool:"Wool",lbl_use:"Use",btn_analyze:"Analyze Sheep",why_breed:"Why this breed?",recent_records:"Recent Records",btn_clear_all:"Clear All",ha_title:"Sheep Health Advisor",ha_sub:"Select symptoms observed in the sheep to get an AI-powered diagnosis",obs_placeholder:"Other Observations (optional)…",btn_analyze_health:"Analyze Health",disease_section:"Detailed Disease Predictions (Probability-Sorted)",score_lbl:"SCORE",prob_chart:"Disease Probability Chart",diet_plan:"Suggested Diet Plan",btn_treated:"Mark as Treated",btn_vet:"Consult Vet",btn_details:"Details",btn_actions:"Actions",no_sheep_title:"No Sheep Detected",no_sheep_msg:"Please upload a clear photo containing a sheep",disclaimer:"Medical Disclaimer: This is a symptom based prediction. Accuracy is limited, always consult a veterinarian for confirmation.",all_records:"All Records",btn_export:"Export CSV",mkt_title:"Mandi Rate Table (PKR/kg)",btn_refresh:"↻ Refresh",eid_msg:"🌙 Eid ul Adha premium active — all rates ×1.6",age_adult:"Adult",age_young:"Young",age_lamb:"Lamb",age_old:"Old",ai_analysis:"AI Analysis",price_calc:"Price Calculator",mandi_formula:"Mandi formula:",weight_lbl:"Live Weight (kg) *",btn_calc:"Calculate Price",price_hint:"Enter weight above and click Calculate",breakdown:"Breakdown",base_rate:"Base Rate/kg",age_factor:"Age Factor",health_factor:"Health Factor",season:"Season",final_rate:"Final Rate/kg",notes_lbl:"Notes",notes_placeholder:"Observations, teeth count, seller info…",add_record:"Add Sheep Record",photo_lbl:"Photo",click_upload:"Click to upload",name_tag:"Name / Tag",price_min:"Price Min (PKR)",price_max:"Price Max (PKR)",err_breed:"Please select a breed.",btn_cancel:"Cancel",btn_save:"Save",health_healthy:"Healthy",health_weak:"Weak",health_diseased:"Diseased",sym_fever:"Fever",sym_coughing:"Coughing",sym_nasal:"Nasal\nDischarge",sym_diarrhea:"Diarrhea",sym_swollen:"Swollen\nBelly",sym_appetite:"Loss of\nAppetite",sym_weakness:"Weakness",sym_limping:"Limping",sym_skin:"Skin\nLesions",sym_wool:"Wool Loss",sym_lwal:"Listlessness",sym_weight:"Weight\nLoss",sym_udder:"Udder\nSwelling",sev_critical:"Critical",sev_high:"High",sev_moderate:"Moderate",sev_alert:"⚠ Severe Condition! Immediate veterinary consultation strongly recommended.",tx_rec:"Treatment Recommendation",no_disease:"No disease patterns detected — sheep appears healthy.",page_dashboard:"Dashboard",page_health:"Health Advisor",page_records:"Records",page_market:"Market Prices",page_price:"Price Calculator",mkt_head_breed:"Breed",mkt_head_rate_h:"Rate/kg (Healthy)",mkt_head_rate_w:"Rate/kg (Weak)",mkt_head_price:"Price @ 40kg",mkt_head_region:"Region",mkt_head_uses:"Uses",no_recs:"No records yet.",weigh_to_price:"Weigh to price",healthy_lbl:"Healthy",unwell_lbl:"Unwell",treat_rec_title:"Treatment recommendation:"},
  ur:{nav_scan:"بھیڑ اسکین",nav_health:"صحت مشیر",nav_records:"ریکارڈز",nav_market:"مارکیٹ قیمت",nav_price:"قیمت کیلکولیٹر",lang_label:"زبان",scan_panel:"بھیڑ تجزیہ پینل",btn_add:"ریکارڈ شامل کریں",img_loaded:"تصویر لوڈ",btn_change:"تبدیل کریں",btn_remove:"ہٹائیں",upload_prompt:"بھیڑ کی تصویر اپلوڈ کریں",btn_browse:"براؤز",lbl_breed:"نسل",lbl_conf:"اعتماد",lbl_health:"صحت",lbl_wool:"اون",lbl_use:"استعمال",btn_analyze:"بھیڑ تجزیہ کریں",why_breed:"یہ نسل کیوں؟",recent_records:"حالیہ ریکارڈز",btn_clear_all:"سب صاف کریں",ha_title:"بھیڑ صحت مشیر",obs_placeholder:"دیگر مشاہدات (اختیاری)…",btn_analyze_health:"صحت تجزیہ",disease_section:"بیماریوں کی پیشگوئی",score_lbl:"اسکور",prob_chart:"بیماری امکان چارٹ",diet_plan:"تجویز کردہ خوراک",btn_treated:"علاج شدہ",btn_vet:"ڈاکٹر سے مشورہ",btn_details:"تفصیل",btn_actions:"اقدامات",no_sheep_title:"کوئی بھیڑ نہیں ملی",no_sheep_msg:"براہ کرم بھیڑ والی واضح تصویر اپلوڈ کریں",disclaimer:"طبی نوٹ: یہ علامات پر مبنی پیشگوئی ہے۔ ہمیشہ ڈاکٹر سے مشورہ کریں۔",all_records:"تمام ریکارڈز",btn_export:"CSV برآمد",mkt_title:"منڈی نرخ (PKR/kg)",btn_refresh:"↻ تازہ کریں",eid_msg:"🌙 عید الاضحی پریمیم فعال — تمام نرخ ×1.6",age_adult:"بالغ",age_young:"جوان",age_lamb:"برا",age_old:"بوڑھا",ai_analysis:"AI تجزیہ",price_calc:"قیمت کیلکولیٹر",mandi_formula:"منڈی فارمولا:",weight_lbl:"زندہ وزن (kg) *",btn_calc:"قیمت حساب کریں",price_hint:"وزن درج کریں",breakdown:"تفصیل",base_rate:"بنیادی نرخ/kg",age_factor:"عمر عنصر",health_factor:"صحت عنصر",season:"موسم",final_rate:"حتمی نرخ/kg",add_record:"بھیڑ ریکارڈ شامل کریں",photo_lbl:"تصویر",click_upload:"اپلوڈ کریں",name_tag:"نام / ٹیگ",price_min:"کم سے کم قیمت (PKR)",price_max:"زیادہ سے زیادہ قیمت (PKR)",err_breed:"نسل منتخب کریں۔",btn_cancel:"منسوخ",btn_save:"محفوظ",health_healthy:"صحت مند",health_weak:"کمزور",health_diseased:"بیمار",sym_fever:"بخار",sym_coughing:"کھانسی",sym_nasal:"ناک\nبہنا",sym_diarrhea:"اسہال",sym_swollen:"پیٹ\nپھولنا",sym_appetite:"بھوک\nنہ لگنا",sym_weakness:"کمزوری",sym_limping:"لنگڑاہٹ",sym_skin:"جلد\nزخم",sym_wool:"اون\nجھڑنا",sym_lwal:"سستی",sym_weight:"وزن\nکمی",sym_udder:"تھن\nسوجن",sev_critical:"بحرانی",sev_high:"زیادہ",sev_moderate:"معتدل",sev_alert:"⚠ سنگین حالت! فوری ڈاکٹر سے رجوع کریں۔",tx_rec:"علاج کی سفارش",no_disease:"کوئی بیماری نہیں ملی — بھیڑ صحت مند لگتی ہے۔",page_dashboard:"ڈیش بورڈ",page_health:"صحت مشیر",page_records:"ریکارڈز",page_market:"مارکیٹ قیمتیں",page_price:"قیمت کیلکولیٹر",mkt_head_breed:"نسل",mkt_head_rate_h:"نرخ/kg (صحت مند)",mkt_head_rate_w:"نرخ/kg (کمزور)",mkt_head_price:"قیمت @ 40kg",mkt_head_region:"علاقہ",mkt_head_uses:"استعمال",no_recs:"کوئی ریکارڈ نہیں۔",weigh_to_price:"وزن کریں",healthy_lbl:"صحت مند",unwell_lbl:"بیمار",treat_rec_title:"علاج:"},
  sd:{nav_scan:"رڍ اسڪين",nav_health:"صحت صلاحڪار",nav_records:"رڪارڊ",nav_market:"مارڪيٽ قيمت",nav_price:"قيمت ڪيلڪيوليٽر",lang_label:"ٻولي",scan_panel:"رڍ تجزيو پينل",btn_add:"رڪارڊ شامل ڪريو",img_loaded:"تصوير لوڊ",btn_change:"تبديل ڪريو",btn_remove:"هٽايو",upload_prompt:"رڍ جي تصوير اپلوڊ ڪريو",btn_browse:"ڳوليو",lbl_breed:"نسل",lbl_conf:"اعتماد",lbl_health:"صحت",lbl_wool:"اون",lbl_use:"استعمال",btn_analyze:"رڍ تجزيو ڪريو",why_breed:"هي نسل ڇو؟",recent_records:"تازا رڪارڊ",btn_clear_all:"سڀ صاف ڪريو",ha_title:"رڍ صحت صلاحڪار",obs_placeholder:"ٻيا مشاهدا (اختياري)…",btn_analyze_health:"صحت تجزيو",disease_section:"بيمارين جي اڳڪٿي",score_lbl:"اسڪور",prob_chart:"بيماري امڪان چارٽ",diet_plan:"تجويز ڪيل خوراڪ",btn_treated:"علاج ٿيل",btn_vet:"ڊاڪٽر صلاح",btn_details:"تفصيل",btn_actions:"اقدام",no_sheep_title:"ڪا رڍ نه ملي",no_sheep_msg:"مهرباني ڪري رڍ واري واضح تصوير اپلوڊ ڪريو",disclaimer:"طبي نوٽ: هي علامتن تي ٻڌل اڳڪٿي آهي. هميشه ڊاڪٽر سان صلاح ڪريو.",all_records:"سڀ رڪارڊ",btn_export:"CSV برآمد",mkt_title:"منڊي نرخ (PKR/kg)",btn_refresh:"↻ تازو ڪريو",eid_msg:"🌙 عيد الاضحيٰ پريميم فعال — سڀ نرخ ×1.6",age_adult:"بالغ",age_young:"جوان",age_lamb:"برو",age_old:"پوڙهو",ai_analysis:"AI تجزيو",price_calc:"قيمت ڪيلڪيوليٽر",mandi_formula:"منڊي فارمولا:",weight_lbl:"زنده وزن (kg) *",btn_calc:"قيمت حساب ڪريو",price_hint:"وزن داخل ڪريو",breakdown:"تفصيل",base_rate:"بنيادي نرخ/kg",age_factor:"عمر عنصر",health_factor:"صحت عنصر",season:"موسم",final_rate:"آخري نرخ/kg",add_record:"رڍ رڪارڊ شامل ڪريو",photo_lbl:"تصوير",click_upload:"اپلوڊ ڪريو",name_tag:"نالو / ٽيگ",price_min:"گهٽ قيمت (PKR)",price_max:"وڌ قيمت (PKR)",err_breed:"نسل چونڊيو۔",btn_cancel:"رد ڪريو",btn_save:"محفوظ ڪريو",health_healthy:"صحتمند",health_weak:"ڪمزور",health_diseased:"بيمار",sym_fever:"بخار",sym_coughing:"کنگهڻ",sym_nasal:"نڪ\nوهڻ",sym_diarrhea:"دست",sym_swollen:"پيٽ\nوڌڻ",sym_appetite:"بک\nنه لڳڻ",sym_weakness:"ڪمزوري",sym_limping:"لنگڙائپ",sym_skin:"چمڙي\nزخم",sym_wool:"اون\nڇڄڻ",sym_lwal:"سستي",sym_weight:"وزن\nگهٽجڻ",sym_udder:"تھڻ\nسوجن",sev_critical:"بحراني",sev_high:"وڌيڪ",sev_moderate:"معتدل",sev_alert:"⚠ سنگين حالت! فوري ڊاڪٽر وٽ وڃو.",tx_rec:"علاج جي سفارش",no_disease:"ڪا بيماري نه ملي.",page_dashboard:"ڊيش بورڊ",page_health:"صحت صلاحڪار",page_records:"رڪارڊ",page_market:"مارڪيٽ قيمتون",page_price:"قيمت ڪيلڪيوليٽر",mkt_head_breed:"نسل",mkt_head_rate_h:"نرخ/kg (صحتمند)",mkt_head_rate_w:"نرخ/kg (ڪمزور)",mkt_head_price:"قيمت @ 40kg",mkt_head_region:"علائقو",mkt_head_uses:"استعمال",no_recs:"ڪو رڪارڊ ناهي.",weigh_to_price:"وزن ڪريو",healthy_lbl:"صحتمند",unwell_lbl:"بيمار",treat_rec_title:"علاج:"}
};
let CL='en';
const T=k=>(LANGS[CL]||LANGS.en)[k]||(LANGS.en[k]||k);
function applyLang(){
  document.body.dir=['ur','sd'].includes(CL)?'rtl':'ltr';
  document.querySelectorAll('[data-key]').forEach(el=>{const t=T(el.getAttribute('data-key'));if(t)el.textContent=t;});
  document.querySelectorAll('[data-placeholder-key]').forEach(el=>{el.placeholder=T(el.getAttribute('data-placeholder-key'))||el.placeholder;});
  buildSymptoms();
}
const SYMS=[
  {id:'fever',sk:'sym_fever',s:`<g stroke="currentColor" stroke-width="1.4" fill="none" stroke-linecap="round"><line x1="12" y1="3" x2="12" y2="13"/><path d="M9 13a3 3 0 106 0 3 3 0 00-6 0z" fill="currentColor" opacity=".25"/><circle cx="12" cy="16" r="3.5"/><line x1="14.5" y1="6" x2="16" y2="6"/><line x1="14.5" y1="8.5" x2="16.5" y2="8.5"/><line x1="14.5" y1="11" x2="16" y2="11"/></g>`},
  {id:'coughing',sk:'sym_coughing',s:`<g stroke="currentColor" stroke-width="1.4" fill="none" stroke-linecap="round"><path d="M7 9c0-3 2-5 5-5s5 2 5 5v2c0 1-.5 2-1 2.5"/><path d="M8 11c0 3 1.5 5 4 5s4-2 4-5"/><path d="M5 14c-1 0-2 1-2 2s1 2 2 2h1M19 14c1 0 2 1 2 2s-1 2-2 2h-1"/><path d="M11 18c0 1 .5 2 1 2s1-1 1-2"/></g>`},
  {id:'nasal discharge',sk:'sym_nasal',s:`<g stroke="currentColor" stroke-width="1.4" fill="none" stroke-linecap="round"><path d="M9 3c0-1 1.5-1 3-1s3 0 3 1l.5 5H8.5z"/><path d="M9 8c0 5 1 10 3 10s3-5 3-10"/><path d="M10 15c.5 2 1 3 2 3s1.5-1 2-3"/></g>`},
  {id:'diarrhea',sk:'sym_diarrhea',s:`<g stroke="currentColor" stroke-width="1.4" fill="none" stroke-linecap="round"><path d="M5 7c2-3 12-3 14 0"/><path d="M6 11c2-2.5 10-2.5 12 0"/><path d="M7 15c2-2 8-2 10 0"/><path d="M12 17v3M10 19l2 1 2-1"/></g>`},
  {id:'swollen belly',sk:'sym_swollen',s:`<g stroke="currentColor" stroke-width="1.4" fill="none" stroke-linecap="round"><path d="M6 16c0-5 2.5-8 6-8s6 3 6 8"/><ellipse cx="12" cy="16" rx="6" ry="4" fill="currentColor" opacity=".1"/><path d="M8 10c1-3 6-3 8 0"/><path d="M9 7c1-1 4-1 6 0"/></g>`},
  {id:'loss of appetite',sk:'sym_appetite',s:`<g stroke="currentColor" stroke-width="1.4" fill="none" stroke-linecap="round"><rect x="4" y="8" width="16" height="10" rx="2"/><path d="M4 12h16M8 8V6M16 8V6"/><line x1="7" y1="17" x2="17" y2="11" stroke-width="1.8"/></g>`},
  {id:'weakness',sk:'sym_weakness',s:`<g stroke="currentColor" stroke-width="1.4" fill="none" stroke-linecap="round"><circle cx="12" cy="6" r="2.8"/><path d="M12 9v4l-2 5M12 13l2 5"/><path d="M8 14l8 0"/><path d="M7 19h10" opacity=".5"/></g>`},
  {id:'limping',sk:'sym_limping',s:`<g stroke="currentColor" stroke-width="1.4" fill="none" stroke-linecap="round"><circle cx="12" cy="5" r="2.5"/><path d="M12 8v6"/><path d="M12 14l-3 6"/><path d="M12 14l2.5 4"/><path d="M9 20h3"/></g>`},
  {id:'skin lesions',sk:'sym_skin',s:`<g stroke="currentColor" stroke-width="1.3" fill="none" stroke-linecap="round"><circle cx="12" cy="12" r="7"/><circle cx="9" cy="9.5" r="1.3" fill="currentColor" opacity=".6"/><circle cx="14.5" cy="8.5" r="1" fill="currentColor" opacity=".6"/><circle cx="14" cy="14" r="1.5" fill="currentColor" opacity=".6"/><circle cx="9.5" cy="14.5" r="1" fill="currentColor" opacity=".5"/></g>`},
  {id:'wool loss',sk:'sym_wool',s:`<g stroke="currentColor" stroke-width="1.4" fill="none" stroke-linecap="round"><path d="M7 19c0-6 2-10 5-10s5 4 5 10"/><path d="M5 15c0-3 1-5 3-6M19 15c0-3-1-5-3-6"/><path d="M9 9l-4-2M15 9l4-2"/></g>`},
  {id:'lwalinig',sk:'sym_lwal',s:`<g stroke="currentColor" stroke-width="1.4" fill="none" stroke-linecap="round"><path d="M12 3c-4.5 0-8 3.5-8 8 0 3 1.5 5.5 4 7l1 3h6l1-3c2.5-1.5 4-4 4-7 0-4.5-3.5-8-8-8z"/><path d="M10 16h4M9 13c1-1 5-1 6 0"/></g>`},
  {id:'weight loss',sk:'sym_weight',s:`<g stroke="currentColor" stroke-width="1.4" fill="none" stroke-linecap="round"><rect x="3" y="11" width="18" height="8" rx="2"/><path d="M7 11V9a5 5 0 0110 0v2"/><path d="M12 15v2.5M10 14l2 1.5 2-1.5"/></g>`},
  {id:'udder swelling',sk:'sym_udder',s:`<g stroke="currentColor" stroke-width="1.4" fill="none" stroke-linecap="round"><path d="M7 13c0-3 2.2-5 5-5s5 2 5 5"/><ellipse cx="12" cy="16.5" rx="5" ry="3.5" fill="currentColor" opacity=".1"/><path d="M8.5 13.5c0 4 1.5 6.5 3.5 6.5s3.5-2.5 3.5-6.5"/><line x1="9.5" y1="16" x2="9.5" y2="19.5"/><line x1="12" y1="16" x2="12" y2="20"/><line x1="14.5" y1="16" x2="14.5" y2="19.5"/></g>`},
];
const DSVG={'Foot Rot':`<g stroke="currentColor" stroke-width="1.4" fill="none" stroke-linecap="round"><path d="M8 15c0-4 2-6 4-7 2-1 4-3 4-5s-1.5-3-4-3-4 2-4 3"/><path d="M10 19c0-2 1-3 2-3s2 1 2 3"/><path d="M9 20h6"/></g>`,'Sheep Pox':`<g stroke="currentColor" stroke-width="1.4" fill="none" stroke-linecap="round"><circle cx="12" cy="10" r="4.5"/><circle cx="7" cy="16" r="2.2"/><circle cx="17" cy="16" r="2.2"/><circle cx="12" cy="19" r="1.5"/></g>`,'Pneumonia':`<g stroke="currentColor" stroke-width="1.4" fill="none" stroke-linecap="round"><path d="M12 4v4"/><path d="M9 8c-2.5 1-4 3.5-4 6s1 4 2.5 4"/><path d="M15 8c2.5 1 4 3.5 4 6s-1 4-2.5 4"/><path d="M8.5 18c1 1.5 2 2 3.5 2s2.5-.5 3.5-2"/></g>`,'Enterotoxemia':`<g stroke="currentColor" stroke-width="1.4" fill="none" stroke-linecap="round"><ellipse cx="12" cy="14" rx="7" ry="5.5"/><path d="M8 9c0-2.5 2-4 4-4s4 1.5 4 4"/><path d="M9 12c1-1.5 6-1.5 6 0"/></g>`,'Parasite Infestation':`<g stroke="currentColor" stroke-width="1.4" fill="none" stroke-linecap="round"><ellipse cx="12" cy="12" rx="7" ry="5.5"/><path d="M9 10c1-1.5 6-1.5 6 0"/><path d="M9 14c1 1.5 6 1.5 6 0"/><circle cx="9.5" cy="12" r="1" fill="currentColor" opacity=".5"/><circle cx="14.5" cy="12" r="1" fill="currentColor" opacity=".5"/></g>`,'Mastitis':`<g stroke="currentColor" stroke-width="1.4" fill="none" stroke-linecap="round"><path d="M7 13c0-3.5 2.2-5.5 5-5.5s5 2 5 5.5"/><ellipse cx="12" cy="17" rx="5" ry="3.5"/><line x1="9.5" y1="17" x2="9.5" y2="20.5"/><line x1="12" y1="17" x2="12" y2="21"/><line x1="14.5" y1="17" x2="14.5" y2="20.5"/></g>`,'Bloat':`<g stroke="currentColor" stroke-width="1.4" fill="none" stroke-linecap="round"><ellipse cx="12" cy="13" rx="8" ry="6.5" fill="currentColor" opacity=".08"/><ellipse cx="12" cy="13" rx="8" ry="6.5"/><path d="M8 10c1.5-2.5 8.5-2.5 8 0"/></g>`,'Anthrax':`<g stroke="currentColor" stroke-width="1.4" fill="none" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l2.5 5.5h5.5l-4.5 3.5 1.5 5.5-5-3-5 3 1.5-5.5-4.5-3.5h5.5z"/></g>`,'Scrapie':`<g stroke="currentColor" stroke-width="1.4" fill="none" stroke-linecap="round"><circle cx="12" cy="7.5" r="3.5"/><path d="M8.5 11c-2.5 1.5-3.5 4-2.5 6.5l1.5 2.5"/><path d="M15.5 11c2.5 1.5 3.5 4 2.5 6.5l-1.5 2.5"/><path d="M10 20h4"/></g>`};
function buildSymptoms(){
  const r1=$('sr1'),r2=$('sr2');if(!r1||!r2)return;
  const sel=new Set(getSel());r1.innerHTML='';r2.innerHTML='';
  SYMS.slice(0,8).forEach(s=>r1.appendChild(mkSB(s,sel.has(s.id))));
  SYMS.slice(8).forEach(s=>r2.appendChild(mkSB(s,sel.has(s.id))));
}
function mkSB(s,on){
  const d=document.createElement('div');
  d.className='sb'+(on?' on':'');d.id='sb-'+s.id.replace(/ /g,'-');
  d.onclick=()=>d.classList.toggle('on');
  d.innerHTML=`<div class="sb-ck">✓</div><div class="sb-ico"><svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">${s.s}</svg></div><div class="sb-lbl">${T(s.sk).replace('\n','<br>')}</div>`;
  return d;
}
function getSel(){return SYMS.filter(s=>{const e=$('sb-'+s.id.replace(/ /g,'-'));return e&&e.classList.contains('on')}).map(s=>s.id);}
window.onload=()=>{buildSymptoms();renderRecs();applyLang();syncBnNav();};
function onDrop(e){e.preventDefault();$('uzone').classList.remove('drag');const f=e.dataTransfer.files[0];if(f?.type.startsWith('image/'))setFile(f);}
function onFile(e){if(e.target.files[0])setFile(e.target.files[0]);}
function setFile(f){
  curFile=f;const u=URL.createObjectURL(f);
  $('uprev').src=u;$('uprev').classList.add('show');
  $('rimg').src=u;show('rimg');hide('rph');
  show('img-actions','flex');hide('uph');
  $('pbadge').classList.add('show');
  hide('rov');hide('feat-grid');hide('reason-box');
  $('abtn').disabled=false;
}
function clearImage(){
  curFile=null;$('finput').value='';$('uprev').src='';$('uprev').classList.remove('show');
  $('pbadge').classList.remove('show');hide('img-actions');show('uph','flex');
  hide('rimg');show('rph','flex');hide('rov');hide('feat-grid');hide('reason-box');
  $('abtn').disabled=true;
}
async function runAnalysis(){
  if(!curFile)return;
  const btn=$('abtn');btn.innerHTML=`<span class="spin"></span>${T('btn_analyze')}…`;btn.disabled=true;
  try{
    const fd=new FormData();fd.append('file',curFile);
    const r=await fetch('/classify',{method:'POST',body:fd,signal:AbortSignal.timeout(30000)});
    if(!r.ok)throw new Error((await r.json().catch(()=>({}))).detail||'Error '+r.status);
    renderResult(await r.json());
  }catch(e){show('rov');$('rov').innerHTML=`<div style="color:var(--re);font-size:11px;padding:4px">⚠ ${e.message}</div>`;}
  btn.textContent=T('btn_analyze');btn.disabled=false;
}
function renderResult(d){
  if(d.is_sheep===false){
    show('rov');
    $('rov').innerHTML=`<div style="text-align:center;padding:12px 8px"><div style="font-size:26px;margin-bottom:7px">🚫</div><div style="color:var(--re);font-size:12px;font-weight:700;margin-bottom:5px">${T('no_sheep_title')}</div><div style="color:var(--tm);font-size:10px;line-height:1.6">${T('no_sheep_msg')}</div></div>`;
    return;
  }
  const cp=Math.round((d.breed_confidence||.5)*100);
  const col=cp>=75?'var(--gr)':cp>=50?'var(--or)':'var(--re)';
  $('ov-breed').textContent=d.breed;$('ov-use').textContent=d.use||'—';
  $('ov-conf').textContent=cp+'%';$('ov-conf').style.color=col;
  const hb=$('ov-hbadge');hb.textContent=cap(d.health);hb.className='hbadge '+d.health;
  $('ov-bars').innerHTML=(d.top_breeds||[]).slice(0,3).map((b,i)=>{
    const p=Math.round(b.probability*100);
    return`<div class="cbar"><div class="cbll"><span>${b.breed}</span><span>${p}%</span></div><div class="cbt"><div class="cbf${i?'':' top'}" style="width:${p}%"></div></div></div>`;
  }).join('');
  const rovEl=$('rov');show('rov');rovEl.classList.remove('show-anim');void rovEl.offsetWidth;rovEl.classList.add('show-anim');
  $('f-breed').textContent=d.breed;$('f-use2').textContent=d.use||'—';
  const fh=$('f-hlth');fh.textContent=cap(d.health);
  fh.style.color=d.health==='healthy'?'var(--gr)':d.health==='weak'?'var(--ye)':'var(--re)';
  $('f-wool').textContent=d.wool||'—';$('f-bcs').textContent=(d.bcs||3)+'/5';
  if($('f-pipe'))$('f-pipe').textContent=d.pipeline||'—';
  show('feat-grid','grid');
  if(d.reason){$('reason-text').textContent=d.reason;show('reason-box');}
  $('mv-breed').textContent=d.breed;$('mv-use').textContent=d.use||'—';
  const mh=$('mv-hlth');mh.textContent=cap(d.health);
  mh.className='mval '+(d.health==='healthy'?'gr':d.health==='weak'?'ye':'re');
  $('mv-conf').textContent=cp+'%';
  window._AI=d;clrPR();
  $('br-ov').value=d.breed||'';$('hl-ov').value=(d.health||'').toLowerCase();
  const save=img=>{recs.unshift({breed:d.breed,use:d.use||'—',health:d.health==='diseased'?'bad':'good',pmin:null,pmax:null,img,ts:Date.now()});srecs();renderRecs();};
  if(curFile){const rd=new FileReader();rd.onload=ev=>save(ev.target.result);rd.readAsDataURL(curFile);}
  else save(null);
}
async function runDiag(){
  const btn=$('ha-btn');btn.innerHTML=`<span class="spin"></span> ${T('btn_analyze_health')}…`;btn.disabled=true;
  try{
    const r=await fetch('/diagnose',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({symptoms:getSel()})});
    if(!r.ok)throw new Error('Server error');
    window._dcIdx=0;
    renderDiag(await r.json());
  }catch(e){alert('Diagnosis failed: '+e.message);}
  btn.textContent=T('btn_analyze_health');btn.disabled=false;
}
function sevLbl(s){return s==='Critical'?T('sev_critical'):s==='High'?T('sev_high'):T('sev_moderate');}
function renderDiag(d){
  const AL=188.5,sc=d.health_score;
  const gc=sc>75?'#d4aa3f':sc>40?'#e8a020':'#e05555';
  [$('garc'),$('garc-glow')].forEach(a=>{if(!a)return;a.style.transition='none';a.style.strokeDasharray=`0 ${AL}`;a.style.stroke=gc;});
  requestAnimationFrame(()=>requestAnimationFrame(()=>{
    [$('garc'),$('garc-glow')].forEach(a=>{if(!a)return;a.style.transition='stroke-dasharray 1.2s cubic-bezier(.4,0,.2,1),stroke .5s ease';a.style.strokeDasharray=`${(sc/100)*AL} ${AL}`;});
  }));
  const gnEl=$('gnum');gnEl.textContent='0';gnEl.style.color=gc;
  gnEl.style.textShadow=`0 0 14px ${gc},0 0 28px ${gc}60`;
  let cur=0;const step=Math.ceil(sc/40);
  const ticker=setInterval(()=>{cur=Math.min(cur+step,sc);gnEl.textContent=cur;if(cur>=sc)clearInterval(ticker);},28);
  $('gstatus').textContent=d.status;$('gstatus').style.color=gc;
  $('gstatus').style.animation='none';void $('gstatus').offsetWidth;$('gstatus').style.animation='fadeUp .4s ease .8s both';$('gstatus').style.opacity='0';setTimeout(()=>$('gstatus').style.opacity='',800);
  show('ha-res');
  const preds=d.predicted_diseases||[];
  $('dcl').innerHTML=preds.slice(0,3).length?preds.slice(0,3).map((x,i)=>{
    window._dcIdx=i;return dcCard(x).replace('<div class="dc"',`<div class="dc" style="animation:scan-in .45s ease ${i*.14}s both"`);
  }).join('')
    :`<div style="padding:14px;text-align:center;color:var(--td);font-size:12px;border:1px dashed var(--bd);border-radius:8px">${T('no_disease')}</div>`;
  setTimeout(()=>preds.slice(0,3).forEach((_,i)=>{const b=$('dcbf'+i);if(b)b.style.width=Math.round(preds[i].probability*100)+'%';}),60);
  $('dcm').innerHTML=preds.slice(3,7).map((x,i)=>{
    const col=SC[x.severity]||'#b8922a',svg=DSVG[x.disease]||`<circle cx="12" cy="12" r="5" fill="none" stroke="currentColor" stroke-width="1.4"/>`;
    return`<div class="mc" style="animation:card-draw .3s ease ${i*.08}s both"><div class="mc-ico"><svg viewBox="0 0 24 24">${svg}</svg></div><div class="mc-name">${x.disease}</div><div class="mc-pct" style="color:${col}">${Math.round(x.probability*100)}%</div></div>`;
  }).join('');
  renderSpark(preds.slice(0,6));show('chart-panel');
  $('diet').innerHTML=d.diet.map(x=>`<div class="diet-item"><div class="d-dot"></div>${x}</div>`).join('');
  show('diet-panel');
  if(preds.length){
    const t=preds[0];
    $('ct').textContent=`${T('tx_rec')}: ${t.disease} (${sevLbl(t.severity)} Severity)`;
    $('cb').innerHTML=`<strong>${T('treat_rec_title')}</strong><ul class="cl">${t.actions.map(a=>`<li>${a}</li>`).join('')}</ul>`;
    show('consult');$('consult').classList.add('on');
  }
}
window._dcIdx=0;
function dcCard(x){
  const i=window._dcIdx++;
  const col=SC[x.severity]||'#b8922a',bg=SBG[x.severity]||'rgba(184,146,42,.1)',bd=SBD[x.severity]||'rgba(184,146,42,.3)';
  const pct=Math.round(x.probability*100),isCrit=(x.severity==='Critical'||x.severity==='High')&&pct>25;
  const svg=DSVG[x.disease]||`<circle cx="12" cy="12" r="5" fill="none" stroke="currentColor" stroke-width="1.4"/>`;
  return`<div class="dc" style="border-left:4px solid ${col}">
<div class="dc-top"><div class="dc-n" style="background:${bg};border:1px solid ${bd};color:${col}">${i+1}</div><div class="dc-icon" style="background:${bg}"><svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" style="color:${col}">${svg}</svg></div>
<div class="dc-body"><div class="dc-name">${x.disease}</div><div class="dc-desc">${x.description}</div>${isCrit?`<div class="sev-alert">${T('sev_alert')}</div>`:''}</div>
<div class="dc-right"><div class="dc-pct" style="color:${col}">${pct}%</div><span class="dc-sev ${x.severity.toLowerCase()}">${sevLbl(x.severity)}</span>
<div class="dc-btns"><button class="dc-tb" onclick="event.stopPropagation();dTab(${i},'details',this)">${T('btn_details')}</button><button class="dc-tb" onclick="event.stopPropagation();dTab(${i},'actions',this)">${T('btn_actions')}</button></div></div></div>
<div class="dc-bar"><div class="dc-bar-f" id="dcbf${i}" style="width:0%;background:${col}"></div></div>
<div class="dc-exp" id="dce${i}" data-x='${JSON.stringify(x).replace(/'/g,"&#39;")}'></div></div>`;
}
function dTab(i,tab,btn){
  const exp=$('dce'+i);if(!exp)return;
  const same=exp.classList.contains('on')&&exp.dataset.t===tab;
  btn.closest('.dc').querySelectorAll('.dc-tb').forEach(b=>b.classList.remove('on'));
  if(same){exp.classList.remove('on');return;}
  btn.classList.add('on');exp.dataset.t=tab;exp.classList.add('on');
  try{
    const x=JSON.parse(exp.dataset.x.replace(/&#39;/g,"'"));
    exp.innerHTML=tab==='details'?`<div style="font-size:11px;color:var(--tm);line-height:1.6">${x.description}</div>`
      :`<div class="act-row">${(x.actions||[]).map(a=>`<span class="act-chip">${a}</span>`).join('')}</div>`;
  }catch{exp.innerHTML='<div style="font-size:11px;color:var(--re)">Error</div>';}
}
function renderSpark(preds){
  const svg=$('spark');if(!preds.length){svg.innerHTML='';return;}
  const[W,H,pL,pR,pT,pB]=[200,68,14,14,14,22];
  const plotW=W-pL-pR,plotH=H-pT-pB;
  const maxP=Math.max(...preds.map(d=>d.probability),.01);
  const pts=preds.map((d,i)=>[Math.round((pL+i/Math.max(preds.length-1,1)*plotW)*10)/10,Math.round((pT+plotH-(d.probability/maxP)*plotH)*10)/10]);
  let line='M '+pts[0].join(',');
  for(let i=1;i<pts.length;i++){const cx=(pts[i-1][0]+pts[i][0])/2;line+=` C ${cx},${pts[i-1][1]} ${cx},${pts[i][1]} ${pts[i].join(',')}`;}
  const area=line+` L ${pts.at(-1)[0]},${H-pB} L ${pts[0][0]},${H-pB} Z`;
  let lineLen=0;for(let i=1;i<pts.length;i++){const dx=pts[i][0]-pts[i-1][0],dy=pts[i][1]-pts[i-1][1];lineLen+=Math.sqrt(dx*dx+dy*dy)*1.2;}
  lineLen=Math.ceil(lineLen)+20;
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
  svg.innerHTML=`<defs><linearGradient id="ag" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#d4aa3f" stop-opacity="0.45"/><stop offset="100%" stop-color="#d4aa3f" stop-opacity="0.02"/></linearGradient><linearGradient id="lg" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#b8922a"/><stop offset="55%" stop-color="#e8c55a"/><stop offset="100%" stop-color="#d4aa3f"/></linearGradient><style>@keyframes draw-line{from{stroke-dashoffset:${lineLen}}to{stroke-dashoffset:0}}@keyframes fade-area{from{opacity:0}to{opacity:1}}@keyframes dot-rise{0%{transform:translateY(10px) scale(0);opacity:0}65%{transform:translateY(-3px) scale(1.3);opacity:1}100%{transform:translateY(0) scale(1);opacity:1}}@keyframes pulse-ring{0%{r:4.5;stroke-opacity:.7}100%{r:11;stroke-opacity:0}}@keyframes lbl-up{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:translateY(0)}}</style></defs>
<line x1="${pL}" y1="${pT}" x2="${W-pR}" y2="${pT}" stroke="#1e1b10" stroke-width="0.5"/>
<line x1="${pL}" y1="${pT+plotH*0.5}" x2="${W-pR}" y2="${pT+plotH*0.5}" stroke="#1e1b10" stroke-width="0.5" stroke-dasharray="2 3"/>
<line x1="${pL}" y1="${H-pB}" x2="${W-pR}" y2="${H-pB}" stroke="#2a2616" stroke-width="0.8"/>
<path d="${area}" fill="url(#ag)" style="animation:fade-area .5s ease .75s both"/>
<path d="${line}" fill="none" stroke="#c9a030" stroke-width="4" stroke-linecap="round" opacity="0.2" style="filter:blur(4px)"/>
<path d="${line}" fill="none" stroke="url(#lg)" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="${lineLen}" stroke-dashoffset="${lineLen}" style="animation:draw-line .75s cubic-bezier(.4,0,.2,1) .1s forwards;filter:drop-shadow(0 0 3px rgba(232,197,90,.8))"/>
${pts.map((p,i)=>{const col=SC[preds[i].severity]||'#d4aa3f';const pct=Math.round(preds[i].probability*100);const d=(0.88+i*0.1).toFixed(2);const d2=(parseFloat(d)+.12).toFixed(2);const nm=preds[i].disease.split(' ')[0];return `<circle cx="${p[0]}" cy="${p[1]}" r="4.5" fill="none" stroke="${col}" stroke-width="1.5" style="animation:pulse-ring 2s ease-out ${d}s infinite"/><g style="animation:dot-rise .4s cubic-bezier(.34,1.56,.64,1) ${d}s both;transform-origin:${p[0]}px ${p[1]}px"><circle cx="${p[0]}" cy="${p[1]}" r="5" fill="${col}" stroke="#09090a" stroke-width="1.5" style="filter:drop-shadow(0 0 5px ${col})"/><circle cx="${p[0]}" cy="${p[1]}" r="1.8" fill="#09090a"/></g><text x="${p[0]}" y="${p[1]-10}" text-anchor="middle" fill="${col}" font-size="8.5" font-family="Space Mono,monospace" font-weight="700" style="animation:lbl-up .3s ease ${d2}s both">${pct}%</text><text x="${p[0]}" y="${H+1}" text-anchor="middle" fill="#5a4a28" font-size="6.5" font-family="Space Grotesk,sans-serif" style="animation:lbl-up .3s ease ${d2}s both">${nm}</text>`;}).join('')}`;
}
function markTx(btn){btn.textContent='✓ '+T('btn_treated');btn.disabled=true;btn.style.color='var(--gr)';btn.style.borderColor='var(--gr)';}
function clrPR(){hide('pres');show('pph');}
async function calcPrice(){
  const w=parseFloat($('w-in').value);
  if(!w||w<=0){$('w-in').style.borderColor='var(--re)';$('w-in').focus();setTimeout(()=>$('w-in').style.borderColor='',1500);return;}
  const ai=window._AI||{};
  try{
    const r=await fetch('/price',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({breed:$('br-ov').value||ai.breed||'Lohi',age:'adult',health:($('hl-ov').value||ai.health||'healthy').toLowerCase(),weight_kg:w})});
    if(!r.ok)throw new Error('Error '+r.status);
    const p=await r.json();
    $('mvp').textContent=p.display;$('mve').textContent='Estimated: PKR '+p.estimated.toLocaleString();
    $('pr-w').textContent=p.weight_kg+' kg';$('pr-b').textContent='PKR '+p.base_rate.toLocaleString()+'/kg';
    $('pr-a').textContent='× '+p.age_factor;$('pr-a').style.color=p.age_factor>=1?'var(--gr)':'var(--ye)';
    $('pr-h').textContent='× '+p.health_factor;$('pr-h').style.color=p.health_factor===1?'var(--gr)':p.health_factor>=.5?'var(--ye)':'var(--re)';
    $('pr-s').textContent=p.eid?'× '+p.eid_factor+' 🌙':'× 1.0';$('pr-s').style.color=p.eid?'var(--or)':'var(--tm)';
    $('pr-r').textContent='PKR '+p.rate_per_kg.toLocaleString()+'/kg';$('pr-f').textContent=p.formula;
    p.eid?$('eidb').classList.add('show'):$('eidb').classList.remove('show');
    hide('pph');show('pres');
    const mp=$('mvp');mp.style.animation='none';void mp.offsetWidth;mp.style.animation='';
    if(recs.length){recs[0].pmin=p.min;recs[0].pmax=p.max;recs[0].weight_kg=p.weight_kg;srecs();renderRecs();}
  }catch(e){alert('Price failed: '+e.message);}
}
async function loadMarket(force=false){
  if(mktData&&!force){rmkt();return;}
  show('mkt-load');hide('mkt-tbl');
  try{
    const r=await fetch('/market-prices',{signal:AbortSignal.timeout(8000)});
    if(!r.ok)throw new Error('Error');
    mktData=await r.json();
    $('mkt-upd').textContent='— '+new Date(mktData.updated).toLocaleTimeString();
    $('eid-banner').classList.toggle('show',mktData.eid);
    rmkt();
  }catch(e){$('mkt-load').innerHTML=`<span style="color:var(--re)">Failed: ${e.message}</span>`;}
}
function mktF(el,age){document.querySelectorAll('.mft').forEach(x=>x.classList.remove('active'));el.classList.add('active');mktAge=age;rmkt();}
function rmkt(){
  if(!mktData)return;hide('mkt-load');show('mkt-tbl','table');
  $('mkt-th').innerHTML=`<th>${T('mkt_head_breed')}</th><th>${T('mkt_head_rate_h')}</th><th>${T('mkt_head_rate_w')}</th><th>${T('mkt_head_price')}</th><th>${T('mkt_head_region')}</th><th>${T('mkt_head_uses')}</th>`;
  $('mtbody').innerHTML=mktData.prices.map(r=>`<tr><td><strong>${r.breed}</strong></td><td class="pr">${r[mktAge+'_healthy_rate'].toLocaleString()}</td><td style="color:var(--ye);font-family:'IBM Plex Mono'">${r[mktAge+'_weak_rate'].toLocaleString()}</td><td class="pr">${r[mktAge+'_healthy_min'].toLocaleString()} – ${r[mktAge+'_healthy_max'].toLocaleString()}</td><td style="color:var(--tm);font-size:11px">${r.region}</td><td style="color:var(--tm)">${r.uses}</td></tr>`).join('');
}
function lrecs(){try{return JSON.parse(localStorage.getItem('sheep_recs')||'[]');}catch{return[];}}
function srecs(){try{localStorage.setItem('sheep_recs',JSON.stringify(recs.slice(0,50)));}catch{}}
function clearAllRecs(){if(!recs.length||!confirm(T('btn_clear_all')+'?'))return;recs=[];srecs();renderRecs();}
function delRec(i){recs.splice(i,1);srecs();renderRecs();}
function renderRecs(){
  $('rec-count').textContent=recs.length?`(${recs.length})`:'';
  $('clear-btn').style.display=recs.length?'inline-flex':'none';
  const empty=`<div style="grid-column:1/-1;padding:18px;text-align:center;color:var(--td);font-size:11.5px;border:1px dashed var(--bd);border-radius:8px">${T('no_recs')}</div>`;
  if(!recs.length){$('recs-mini').innerHTML=$('recs-large').innerHTML=empty;return;}
  const th=r=>r.img?`<img src="${r.img}" style="width:44px;height:44px;object-fit:cover;border-radius:6px" alt="">`:' 🐑';
  const ps=r=>r.pmin&&r.pmax?r.pmin.toLocaleString()+' – '+r.pmax.toLocaleString():T('weigh_to_price');
  $('recs-mini').innerHTML=recs.slice(0,4).map((r,i)=>
    `<div class="rc" style="animation:card-draw .3s ease ${i*.07}s both"><div class="rthumb">${th(r)}</div><div style="flex:1"><div style="font-size:10px;color:var(--tm)">${T('lbl_breed')}</div><div style="font-size:12px;font-weight:600">${r.breed}</div><div style="font-size:10px;color:var(--tm)">${T('lbl_use')}: ${r.use||'—'}</div></div><div class="rsb ${r.health}">${r.health==='good'?'✓':'!'}</div><div style="text-align:right"><div style="font-size:10px;color:var(--tm)">PKR</div><div style="font-size:11px;font-weight:700;color:${r.pmin?'var(--or)':'var(--td)'}">${ps(r)}</div></div><button class="del-btn" onclick="event.stopPropagation();delRec(${i})">✕</button></div>`
  ).join('');
  $('recs-large').innerHTML=recs.map((r,i)=>
    `<div class="rlc" style="animation:page-fan .32s ease ${i*.06}s both"><div class="rlci">${r.img?`<img src="${r.img}" alt="">`:''}</div><div class="rlcb"><div class="rlcbr">${r.breed}</div><div class="rlcm"><span>${r.use||'—'}</span><span style="color:${r.health==='good'?'var(--gr)':'var(--re)'}">${r.health==='good'?T('healthy_lbl'):T('unwell_lbl')}</span></div><div class="rlcp">${r.pmin&&r.pmax?r.pmin.toLocaleString()+' – '+r.pmax.toLocaleString()+' PKR':T('weigh_to_price')}</div></div><button class="rlc-del" onclick="event.stopPropagation();delRec(${i})">✕</button></div>`
  ).join('');
}
function exportCSV(){
  if(!recs.length){alert(T('no_recs'));return;}
  const csv=[['Breed','Use','Health','WeightKg','PriceMin','PriceMax','Timestamp'],...recs.map(r=>[r.breed,r.use||'—',r.health,r.weight_kg||'',r.pmin||'',r.pmax||'',new Date(r.ts).toISOString()])].map(r=>r.join(',')).join('\n');
  const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));a.download='sheep_records.csv';a.click();
}
function openModal(){
  modalImg=null;
  ['r-name','r-weight','r-pmin','r-pmax','r-notes'].forEach(id=>$(id).value='');
  $('r-breed').value='';
  const prev=$('mimg-prev');if(prev){prev.src='';prev.style.display='none';}
  const ph=$('mimg-ph');if(ph)ph.style.display='';
  hide('r-err');
  $('modal').classList.add('open');
}
function closeModal(){$('modal').classList.remove('open');}
function prevMI(e){const f=e.target.files[0];if(!f)return;const rd=new FileReader();rd.onload=ev=>{modalImg=ev.target.result;$('mimg-prev').src=modalImg;show('mimg-prev');hide('mimg-ph');};rd.readAsDataURL(f);}
function submitRec(){
  const breed=$('r-breed').value;if(!breed){show('r-err');return;}
  recs.unshift({name:$('r-name').value||null,breed,use:$('r-use').value,weight_kg:parseFloat($('r-weight').value)||null,pmin:parseInt($('r-pmin').value)||null,pmax:parseInt($('r-pmax').value)||null,health:$('r-hlth').value,notes:$('r-notes').value||null,img:modalImg,ts:Date.now()});
  srecs();renderRecs();closeModal();
}
window.addEventListener('load',()=>{
  document.querySelectorAll('.panel').forEach((p,i)=>{p.style.animationDelay=(i*0.09)+'s';});
  document.querySelectorAll('.ni').forEach((n,i)=>{n.style.animation=`ledger-entry .32s ease ${(i*0.07+0.05).toFixed(2)}s both`;});
});
const VT={scan:'page_dashboard',health:'page_health',records:'page_records',market:'page_market',price:'page_price'};
function sv(v,el){
  document.querySelectorAll('.view').forEach(x=>x.classList.remove('active'));
  document.querySelectorAll('.ni,.bn-item').forEach(x=>x.classList.remove('active'));
  const view=$('view-'+v);if(view)view.classList.add('active');
  if(el)el.classList.add('active');
  const bn=document.querySelector('.bn-item[data-view="'+v+'"]');
  if(bn)bn.classList.add('active');
  const ni=document.querySelector('.ni[onclick*="\''+v+'\'"]');
  if(ni)ni.classList.add('active');
  const pt=$('page-title');if(pt)pt.textContent=T(VT[v]);
  const rs=$('rsidebar');
  if(rs)rs.style.display=(v==='scan'&&window.innerWidth>768)?'flex':'none';
  if(v==='market')loadMarket();
  if(v==='price')clrPR();
}
function syncBnNav(){
  const active=document.querySelector('.view.active');
  if(!active)return;
  const v=active.id.replace('view-','');
  document.querySelectorAll('.bn-item').forEach(b=>b.classList.remove('active'));
  const bn=document.querySelector('.bn-item[data-view="'+v+'"]');
  if(bn)bn.classList.add('active');
  document.querySelectorAll('.bn-item[data-view]').forEach(btn=>{
    const key='nav_'+btn.getAttribute('data-view');
    const span=btn.querySelector('span');
    if(span&&T(key))span.textContent=T(key);
  });
}
function setLang(el,lang){
  document.querySelectorAll('.lo').forEach(l=>l.classList.remove('active'));
  el.classList.add('active');CL=lang;applyLang();
  const vid=document.querySelector('.view.active')?.id.replace('view-','');
  if(vid){const pt=$('page-title');if(pt)pt.textContent=T(VT[vid]||'page_dashboard');}
  if(mktData)rmkt();
  syncBnNav();
}
window.addEventListener('resize',()=>{
  const rs=$('rsidebar');
  if(!rs)return;
  const scanActive=$('view-scan')&&$('view-scan').classList.contains('active');
  rs.style.display=(scanActive&&window.innerWidth>768)?'flex':'none';
});
(function(){
const cv=document.getElementById('bg-canvas');
if(!cv)return;
const ctx=cv.getContext('2d',{alpha:false});
let W=0,H=0,t=0;
function resize(){W=cv.width=window.innerWidth;H=cv.height=window.innerHeight;initGrass();initStars();}
window.addEventListener('resize',()=>{resize();});
resize();
const SKY=[{r:5,g:3,b:8},{r:8,g:5,b:14},{r:14,g:8,b:12},{r:22,g:11,b:6},{r:35,g:18,b:5},{r:24,g:12,b:4},{r:10,g:6,b:8},{r:5,g:3,b:8}];
function skyAt(p){p=((p%1)+1)%1;let lo=Math.floor(p*(SKY.length-1)),hi=lo+1;const f=p*(SKY.length-1)-lo;if(hi>=SKY.length)hi=SKY.length-1;const A=SKY[lo],B=SKY[hi];return{r:A.r+(B.r-A.r)*f,g:A.g+(B.g-A.g)*f,b:A.b+(B.b-A.b)*f};}
let stars=[];
function initStars(){stars=Array.from({length:90},()=>({x:Math.random()*W,y:Math.random()*H*0.55,r:0.3+Math.random()*1.0,twinkle:Math.random()*Math.PI*2,speed:0.01+Math.random()*0.03}));}
function hillY(x,seed,amp,freq,yBase){let y=yBase;y+=Math.sin(x*freq+seed)*amp;y+=Math.sin(x*freq*1.7+seed*2.3)*amp*0.45;y+=Math.sin(x*freq*3.1+seed*0.7)*amp*0.22;return y;}
let blades=[];
function initGrass(){blades=Array.from({length:Math.floor(W/9)},(_,i)=>({x:(i+Math.random()*0.6)/Math.floor(W/9)*W,h:0.05+Math.random()*0.07,w:0.6+Math.random()*1.1,phase:Math.random()*Math.PI*2,spd:0.004+Math.random()*0.006,sway:3+Math.random()*7,stiff:0.5+Math.random()*0.5,tint:Math.random()}));}
let windPhase=0;
function windStr(t){return 0.4+0.6*Math.abs(Math.sin(t*0.0003))*Math.abs(Math.sin(t*0.00071));}
const FLOCK=7;
const sheep=Array.from({length:FLOCK},(_,i)=>({x:W*(0.15+i*0.1+Math.random()*0.08),y:0,vx:(Math.random()-.5)*0.25,vy:(Math.random()-.5)*0.1,angle:Math.random()*Math.PI*2,size:4+Math.random()*4,restTimer:Math.floor(Math.random()*300),resting:Math.random()<0.4,restDur:0,woolPhase:Math.random()*Math.PI*2,tailPhase:Math.random()*Math.PI*2,headBob:Math.random()*Math.PI*2,wanderStrength:0.5+Math.random()*1.0,turnSpeed:0.005+Math.random()*0.02}));
function groundY(x){return hillY(x/W,1.2,H*0.038,2.8,H*0.72);}
function updateSheep(){
  sheep.forEach((s,i)=>{
    s.woolPhase+=0.018;s.tailPhase+=0.04;s.headBob+=0.022;
    const gY=groundY(s.x);s.y+=(gY-s.y)*0.08;
    if(s.resting){s.restDur--;if(s.restDur<=0){s.resting=false;s.restTimer=100+Math.floor(Math.random()*400);}return;}
    s.restTimer--;
    if(s.restTimer<=0){s.resting=true;s.restDur=60+Math.floor(Math.random()*180);s.restTimer=120+Math.floor(Math.random()*350);}
    let cx=0,cy=0;sheep.forEach(o=>{cx+=o.x;cy+=o.y;});cx/=FLOCK;cy/=FLOCK;
    const fcx=cx-s.x,fcy=cy-s.y,fd=Math.sqrt(fcx*fcx+fcy*fcy)||1;
    if(fd>W*0.18){const fa=Math.atan2(fcy,fcx);let da=fa-s.angle;if(da>Math.PI)da-=Math.PI*2;if(da<-Math.PI)da+=Math.PI*2;s.angle+=da*0.008;}
    sheep.forEach((o,j)=>{if(i===j)return;const dx=s.x-o.x,dy=s.y-o.y,d=Math.sqrt(dx*dx+dy*dy)||1;if(d<s.size*5+o.size*5)s.angle+=Math.atan2(dy,dx)*0.06;});
    const wx=windStr(t)*Math.cos(windPhase);s.angle+=wx*0.002*s.wanderStrength;
    s.angle+=(Math.random()-.5)*s.turnSpeed*s.wanderStrength;
    s.angle=Math.max(-Math.PI*0.15,Math.min(Math.PI*1.15,s.angle));
    const spd=0.2+s.size*0.04;s.x+=Math.cos(s.angle)*spd;s.y+=Math.sin(s.angle)*spd*0.3;
    if(s.x<-40)s.x=W+40;if(s.x>W+40)s.x=-40;
    s.y=Math.max(H*0.65,Math.min(H*0.82,s.y));
  });
}
function drawSheep(s,sa){
  const sc=s.size,pulse=0.94+0.06*Math.sin(s.woolPhase),r=sc*pulse;
  const facing=Math.cos(s.angle)>0?1:-1,bobY=s.resting?0:Math.sin(s.headBob)*1.2;
  ctx.save();ctx.translate(s.x,s.y);
  const ag=ctx.createRadialGradient(0,0,r,0,0,r*3.5);
  ag.addColorStop(0,`rgba(210,165,80,${0.06*(1-sa*0.5)})`);ag.addColorStop(1,'rgba(210,165,80,0)');
  ctx.fillStyle=ag;ctx.beginPath();ctx.arc(0,0,r*3.5,0,Math.PI*2);ctx.fill();
  ctx.save();ctx.scale(1,0.22);ctx.fillStyle='rgba(0,0,0,0.18)';ctx.beginPath();ctx.ellipse(0,r*4.5,r*1.8,r*0.9,0,0,Math.PI*2);ctx.fill();ctx.restore();
  const wc=`rgba(218,200,158,${0.72+sa*0.22})`,wd=`rgba(175,155,110,${0.65+sa*0.2})`;
  ctx.fillStyle=wd;ctx.beginPath();ctx.ellipse(0,0,r*1.4,r,0,0,Math.PI*2);ctx.fill();
  [[-.7,-.55],[0,-.75],[.7,-.5],[-.4,.1],[.4,.05]].forEach(([ox,oy])=>{ctx.fillStyle=wc;ctx.beginPath();ctx.arc(ox*r,oy*r,r*0.72,0,Math.PI*2);ctx.fill();});
  const hx=facing*r*1.55,hy=-r*0.18+bobY;
  ctx.fillStyle=`rgba(175,148,95,${0.82+sa*0.15})`;ctx.beginPath();ctx.ellipse(hx,hy,r*0.52,r*0.44,facing*0.12,0,Math.PI*2);ctx.fill();
  ctx.fillStyle=`rgba(155,125,75,.75)`;ctx.beginPath();ctx.ellipse(hx+facing*r*0.18,hy-r*0.42,r*0.18,r*0.28,-0.3*facing,0,Math.PI*2);ctx.fill();
  const ex=hx+facing*r*0.22,ey=hy-r*0.08;
  ctx.fillStyle=`rgba(235,210,140,.9)`;ctx.beginPath();ctx.arc(ex,ey,r*0.14,0,Math.PI*2);ctx.fill();
  ctx.fillStyle='rgba(15,10,5,.9)';ctx.beginPath();ctx.arc(ex+facing*r*0.04,ey,r*0.07,0,Math.PI*2);ctx.fill();
  const la=s.resting?0:Math.sin(s.woolPhase*2)*0.4;
  [[-0.6,0.2],[0.6,0.2],[-0.15,0.3],[0.15,0.3]].forEach(([lx],li)=>{
    const sw=(li%2===0?1:-1)*la;ctx.strokeStyle=`rgba(150,120,70,${0.8+sa*0.15})`;ctx.lineWidth=r*0.28;ctx.lineCap='round';
    ctx.beginPath();ctx.moveTo(lx*r,r*0.7);ctx.lineTo(lx*r+sw*2,r*1.7);ctx.stroke();
  });
  const tw=Math.sin(s.tailPhase)*8;ctx.strokeStyle=`rgba(200,178,130,.7)`;ctx.lineWidth=r*0.4;
  ctx.beginPath();ctx.moveTo(-facing*r*1.2,-r*0.1);ctx.quadraticCurveTo(-facing*r*1.6+tw,-r*0.5,-facing*r*1.55,-r*0.1);ctx.stroke();
  ctx.restore();
}
const lantern={x:0.72,y:0.69,phase:0,spd:0.008,drift:0,driftSpd:0.0004};
function draw(){
  if(window.innerWidth<=768){requestAnimationFrame(draw);return;}
  t++;windPhase+=0.0007;lantern.phase+=lantern.spd;lantern.drift=Math.sin(t*lantern.driftSpd)*W*0.015;
  const dawn=((t*0.00035)%1),sky=skyAt(dawn),skyBright=Math.max(dawn<0.5?dawn*2:(1-dawn)*2,0.0);
  const{r:sr,g:sg,b:sb}=sky;
  const skyGrad=ctx.createLinearGradient(0,0,0,H);
  skyGrad.addColorStop(0,`rgb(${sr},${sg},${sb})`);
  const hr=Math.min(255,sr+18+skyBright*40),hg=Math.min(255,sg+10+skyBright*22),hb=Math.min(255,sb+4+skyBright*8);
  skyGrad.addColorStop(0.55,`rgb(${hr|0},${hg|0},${hb|0})`);
  skyGrad.addColorStop(1,`rgb(${Math.min(255,sr+6)|0},${sg|0},${sb|0})`);
  ctx.fillStyle=skyGrad;ctx.fillRect(0,0,W,H);
  const starAlpha=Math.max(0,1-skyBright*3.5);
  if(starAlpha>0.01){stars.forEach(s=>{s.twinkle+=s.speed;const ta=starAlpha*(0.4+0.6*Math.abs(Math.sin(s.twinkle)));ctx.beginPath();ctx.arc(s.x,s.y,s.r,0,Math.PI*2);ctx.fillStyle=`rgba(255,240,200,${ta})`;ctx.fill();});}
  const moonX=W*(0.15+dawn*0.55),moonY=H*(0.08+dawn*0.18),moonAlpha=Math.max(0,1-skyBright*4);
  if(moonAlpha>0.01){const mg=ctx.createRadialGradient(moonX,moonY,0,moonX,moonY,45);mg.addColorStop(0,`rgba(240,220,170,${moonAlpha*0.18})`);mg.addColorStop(1,'rgba(240,220,170,0)');ctx.fillStyle=mg;ctx.beginPath();ctx.arc(moonX,moonY,45,0,Math.PI*2);ctx.fill();ctx.fillStyle=`rgba(238,225,185,${moonAlpha*0.92})`;ctx.beginPath();ctx.arc(moonX,moonY,14,0,Math.PI*2);ctx.fill();ctx.fillStyle=`rgba(${sr+4},${sg+2},${sb},${moonAlpha*0.88})`;ctx.beginPath();ctx.arc(moonX+6,moonY-2,12,0,Math.PI*2);ctx.fill();}
  if(skyBright>0.05){const sg2=ctx.createRadialGradient(W*0.5,H*0.58,0,W*0.5,H*0.58,H*0.45);sg2.addColorStop(0,`rgba(200,130,30,${skyBright*0.22})`);sg2.addColorStop(0.35,`rgba(180,100,20,${skyBright*0.12})`);sg2.addColorStop(1,'rgba(180,100,20,0)');ctx.fillStyle=sg2;ctx.fillRect(0,0,W,H);}
  [{seed:0.8,amp:0.09,freq:1.8,yBase:0.58,dark:0.82},{seed:2.1,amp:0.06,freq:2.4,yBase:0.64,dark:0.90},{seed:3.7,amp:0.04,freq:3.2,yBase:0.70,dark:0.96}].forEach(({seed,amp,freq,yBase,dark})=>{
    const hd=Math.max(0,dark-skyBright*0.15);ctx.beginPath();ctx.moveTo(0,H);
    for(let x=0;x<=W;x+=4)ctx.lineTo(x,hillY(x/W,seed,H*amp,freq,H*yBase));
    ctx.lineTo(W,H);ctx.closePath();ctx.fillStyle=`rgba(${(10+skyBright*25)|0},${(6+skyBright*14)|0},${(2+skyBright*4)|0},${hd})`;ctx.fill();
  });
  const wind=windStr(t),windDir=Math.sin(windPhase)*0.8;
  blades.forEach(b=>{
    b.phase+=b.spd;const bh=b.h*H,bx=b.x,by=groundY(bx);
    const sway=(Math.sin(b.phase+bx*0.008)*b.sway+windDir*b.sway*wind*(1-b.stiff))*1.4;
    if(by>H*0.85)return;
    const rc=`rgba(${(60+b.tint*30)|0},${(38+b.tint*22)|0},${(5+b.tint*8)|0},0.6)`;
    const tc=`rgba(${(140+b.tint*50+skyBright*40)|0},${(95+b.tint*35+skyBright*20)|0},${(12+b.tint*15)|0},0.7)`;
    const g=ctx.createLinearGradient(bx,by,bx+sway,by-bh);g.addColorStop(0,rc);g.addColorStop(1,tc);
    ctx.beginPath();ctx.moveTo(bx,by);ctx.quadraticCurveTo(bx+sway*0.5,by-bh*0.55,bx+sway,by-bh);
    ctx.strokeStyle=g;ctx.lineWidth=b.w;ctx.lineCap='round';ctx.stroke();
  });
  updateSheep();const sorted=[...sheep].sort((a,b)=>a.y-b.y);sorted.forEach(s=>drawSheep(s,skyBright));
  const lx=W*lantern.x+lantern.drift,ly=groundY(lx)-8,la=0.7-skyBright*0.5;
  if(la>0.05){const li=0.7+0.3*Math.sin(lantern.phase);const lb=ctx.createRadialGradient(lx,ly,0,lx,ly,60);lb.addColorStop(0,`rgba(220,150,40,${la*li*0.22})`);lb.addColorStop(0.4,`rgba(180,100,20,${la*li*0.10})`);lb.addColorStop(1,'rgba(180,100,20,0)');ctx.fillStyle=lb;ctx.beginPath();ctx.arc(lx,ly,60,0,Math.PI*2);ctx.fill();const lc=ctx.createRadialGradient(lx,ly,0,lx,ly,8);lc.addColorStop(0,`rgba(255,200,80,${la*li*0.9})`);lc.addColorStop(1,'rgba(255,180,40,0)');ctx.fillStyle=lc;ctx.beginPath();ctx.arc(lx,ly,8,0,Math.PI*2);ctx.fill();ctx.fillStyle=`rgba(255,240,180,${la*li})`;ctx.beginPath();ctx.arc(lx,ly,2.5,0,Math.PI*2);ctx.fill();}
  const haze=ctx.createLinearGradient(0,H*0.55,0,H*0.75);haze.addColorStop(0,'rgba(0,0,0,0)');haze.addColorStop(0.4,`rgba(${(sr+8)|0},${(sg+4)|0},${sb},0.18)`);haze.addColorStop(1,'rgba(0,0,0,0)');ctx.fillStyle=haze;ctx.fillRect(0,H*0.55,W,H*0.2);
  const vig=ctx.createRadialGradient(W*0.5,H*0.44,H*0.05,W*0.5,H*0.5,Math.max(W,H)*0.72);vig.addColorStop(0,'rgba(0,0,0,0)');vig.addColorStop(0.4,'rgba(0,0,0,0.18)');vig.addColorStop(1,'rgba(0,0,0,0.92)');ctx.fillStyle=vig;ctx.fillRect(0,0,W,H);
  requestAnimationFrame(draw);
}
draw();
})();
</script>

<style>
.bottom-nav{
  display:none;
  position:fixed;bottom:0;left:0;right:0;
  height:62px;
  padding-bottom:env(safe-area-inset-bottom,0px);
  background:rgba(9,9,10,.97);
  border-top:1px solid var(--bd);
  backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);
  z-index:500;
  align-items:stretch;justify-content:space-around;
}
.bn-item{
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:3px;flex:1;cursor:pointer;padding:7px 2px 5px;
  color:var(--tm);font-size:8.5px;font-weight:600;
  font-family:var(--font-body);letter-spacing:.04em;text-transform:uppercase;
  transition:color .18s;border:none;background:none;
  -webkit-tap-highlight-color:transparent;outline:none;position:relative;
}
.bn-item.active{color:var(--orl)}
.bn-item.active::after{
  content:'';position:absolute;top:0;left:20%;right:20%;
  height:2px;background:linear-gradient(90deg,transparent,var(--orl),transparent);
}
.bn-item svg{width:22px;height:22px;flex-shrink:0}
.bn-item.active svg{filter:drop-shadow(0 0 5px rgba(232,197,90,.55))}
@media(max-width:768px){
  .bottom-nav{display:flex !important}
  .sidebar{display:none !important}
  .rsidebar{display:none !important}
  .main{width:100% !important}
  .content{flex-direction:column;overflow:hidden}
  .topbar{height:52px !important;padding:0 14px !important}
  .topbar h1{font-size:16px !important;line-height:52px !important}
  .center{
    padding:12px 10px !important;
    padding-bottom:calc(70px + env(safe-area-inset-bottom,8px)) !important;
    gap:10px !important;
    overflow-y:auto;
    -webkit-overflow-scrolling:touch;
    height:calc(100vh - 52px);
    max-height:calc(100vh - 52px);
  }
  .grid2{display:flex !important;flex-direction:column !important;gap:10px !important}
  .upload-zone{min-height:160px !important;padding:18px 12px !important}
  .result-area{min-height:160px !important;width:100% !important}
  .rov{position:relative !important;top:auto !important;right:auto !important;width:100% !important;min-width:0 !important;display:none;border-radius:0 0 8px 8px !important}
  .rov.show-anim{display:block !important}
  .analyze-btn{padding:14px !important;font-size:17px !important}
  .feat-grid{grid-template-columns:1fr 1fr !important}
  .rlg{grid-template-columns:1fr 1fr !important}
  .recs-grid{grid-template-columns:1fr !important}
  .rlci{height:70px !important}
  .ha-title{font-size:22px !important}
  .ha-results-wrap{flex-direction:column !important}
  .ha-results-right{width:100% !important;min-width:0 !important}
  .sym-grid{grid-template-columns:repeat(4,1fr) !important}
  .sym-grid2{grid-template-columns:repeat(4,1fr) !important}
  .sb{min-height:66px;padding:8px 4px 6px}
  .sb-ico{width:24px !important;height:24px !important}
  .sb-lbl{font-size:7.5px !important}
  .price-layout{flex-direction:column !important}
  .price-form{width:100% !important}
  .modal{width:95vw !important;max-width:95vw !important}
  .frow{grid-template-columns:1fr !important;gap:8px !important}
  .mtable-wrap{overflow-x:auto;-webkit-overflow-scrolling:touch}
  .mtable{min-width:460px;font-size:11px}
  .gauge-wrap{width:130px !important;height:86px !important}
  .gauge-wrap svg{width:130px !important;height:86px !important}
  .gn{font-size:28px !important}
  .panel{padding:11px 10px !important}
  .ph{flex-wrap:wrap;gap:6px;min-height:auto !important}
  .btn,.btn2,.btn-danger{height:38px !important;min-height:38px}
  .mft{height:30px;font-size:9px;padding:0 9px}
  .dc-top{gap:7px;padding:9px 10px}
  .dc-name{font-size:12px}
  .dc-pct{font-size:18px !important}
  .dc-icon{width:34px !important;height:34px !important}
  .mini-grid{grid-template-columns:1fr 1fr !important}
  .eid-banner{font-size:10px}
  #bg-canvas{display:none !important}
  body{background:var(--bg) !important}
}
@media(max-width:420px){
  .rlg{grid-template-columns:1fr !important}
  .sym-grid{grid-template-columns:repeat(3,1fr) !important}
  .sym-grid2{grid-template-columns:repeat(3,1fr) !important}
  .feat-grid{grid-template-columns:1fr !important}
}
</style>
</body>
</html>"""

# ── FastAPI app ───────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("="*60)
    log.info("AI Sheep Analyzer — Production Hybrid Mode v2.0")
    log.info(f"  YOLO:          {'✓ YOLOv8x' if YOLO_MODEL  else '✗ pixel fallback'}")
    log.info(f"  Custom Model:  {'✓ EfficientNetV2-ONNX' if CUSTOM_MODEL else '✗ not loaded (run train_custom.py)'}")
    log.info(f"  CLIP-1:        {'✓ ViT-L/14@336px' if CLIP_MODEL  else '✗ pixel fallback'}")
    log.info(f"  CLIP-2:        {'✓ ViT-L/14' if CLIP_MODEL_2 else '✗ not loaded'}")
    log.info(f"  BioBERT:       {'✓ loaded' if VET_MODEL  else '✗ dict fallback'}")
    log.info(f"  Device:        {DEVICE}")
    log.info(f"  Prometheus:    {'✓ enabled' if PROMETHEUS_AVAILABLE else '✗ disabled'}")
    log.info(f"  MLflow:        {'✓ enabled' if MLFLOW_AVAILABLE else '✗ disabled'}")
    log.info("="*60)
    log.info("Pipeline: YOLO → Custom Model → CLIP Ensemble → Pixel Fallback")
    if CLIP_MODEL:
        try:
            classify(Image.new("RGB",(336,336),(128,128,128)))
            log.info("✓ Warmup done")
        except Exception as e:
            log.warning(f"Warmup failed: {e}")
    yield

app = FastAPI(title="AI Sheep Analyzer", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(FRONTEND_HTML)

@app.get("/metrics")
def metrics():
    if not PROMETHEUS_AVAILABLE:
        raise HTTPException(503, "Prometheus not installed — pip install prometheus-client")
    return FastAPIResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
def health_check():
    return {
        "status":       "ok",
        "cv2":          CV2_AVAILABLE,
        "yolo":         YOLO_MODEL is not None,
        "custom_model": CUSTOM_MODEL is not None,
        "clip":         CLIP_MODEL is not None,
        "clip2":        CLIP_MODEL_2 is not None,
        "vet_model":    VET_MODEL is not None,
        "device":       DEVICE,
        "prometheus":   PROMETHEUS_AVAILABLE,
        "mlflow":       MLFLOW_AVAILABLE,
        "pipeline":     "YOLO → Custom → CLIP → Pixel",
    }

@app.get("/breeds")
def get_breeds():
    return {"breeds":[{"name":b,"base_rate_kg":BREED_RATE[b]} for b in BREEDS]}

@app.get("/market-prices")
def get_market_prices():
    return JSONResponse(market_prices())

@app.post("/price")
async def calc_price(data: dict):
    try:    w = float(data.get("weight_kg",0))
    except: w = 0
    if w <= 0: raise HTTPException(400, "weight_kg must be positive")
    try:    return JSONResponse(price(str(data.get("breed","Lohi")), str(data.get("age","adult")),
                                      str(data.get("health","healthy")), w))
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/diagnose")
async def do_diagnose(data: dict):
    try:
        result = diagnose(data.get("symptoms",[]))
        if PROMETHEUS_AVAILABLE:
            DIAGNOSE_TOTAL.labels(status=result.get("status","unknown")).inc()
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/classify")
async def do_classify(file: UploadFile = File(...)):
    ct = (file.content_type or "").lower().split(";")[0].strip()
    if ct and not ct.startswith("image/"): raise HTTPException(415, f"Expected image, got: {ct}")
    raw = await file.read()
    if len(raw) > MAX_UPLOAD: raise HTTPException(413, "File too large.")
    try:    img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e: raise HTTPException(400, f"Bad image: {e}")
    try:
        if PROMETHEUS_AVAILABLE: ACTIVE_REQUESTS.inc()
        t0 = time.perf_counter()

        with (PREDICTION_LATENCY.time() if PROMETHEUS_AVAILABLE else _nullctx()):
            result = classify(img)

        elapsed = round((time.perf_counter()-t0)*1000)
        result.pop("_price_breed", None)

        if PROMETHEUS_AVAILABLE and result.get("is_sheep"):
            conf   = result.get("breed_confidence", 0)
            bucket = "high" if conf >= 0.75 else "mid" if conf >= 0.50 else "low"
            PREDICTIONS_TOTAL.labels(
                breed=result.get("breed","unknown"),
                confidence_bucket=bucket,
                pipeline=result.get("pipeline","unknown")[:30],
            ).inc()
            CONFIDENCE_HISTOGRAM.observe(conf)
            _update_drift(conf)

        if MLFLOW_AVAILABLE and result.get("is_sheep"):
            try:
                with mlflow.start_run(run_name="classify_request", nested=True):
                    mlflow.log_metrics({
                        "breed_confidence":  result.get("breed_confidence", 0),
                        "age_confidence":    result.get("age_confidence", 0),
                        "health_confidence": result.get("health_confidence", 0),
                        "elapsed_ms":        elapsed,
                    })
                    mlflow.log_params({
                        "breed":    result.get("breed","unknown"),
                        "pipeline": result.get("pipeline","unknown")[:50],
                    })
            except Exception:
                pass

        return JSONResponse({**result, "quality": image_quality(img), "elapsed_ms": elapsed})
    except Exception as e:
        log.error(traceback.format_exc())
        raise HTTPException(500, f"Classification failed: {e}")
    finally:
        if PROMETHEUS_AVAILABLE: ACTIVE_REQUESTS.dec()

from contextlib import contextmanager
@contextmanager
def _nullctx():
    yield

if __name__ == "__main__":
    import uvicorn
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=5050)
    args = p.parse_args()
    print(f"\n{'='*60}")
    print(f"  AI Sheep Analyzer v2.0 — Production ML Pipeline")
    print(f"  http://localhost:{args.port}")
    print(f"  Metrics: http://localhost:{args.port}/metrics")
    print(f"  Pipeline: YOLO → Custom Model → CLIP → Pixel Fallback")
    print(f"{'='*60}\n")
    uvicorn.run(app, host="0.0.0.0", port=args.port, reload=False)
