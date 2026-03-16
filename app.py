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

# ── Frontend HTML (unchanged) ─────────────────────────────────────
_HTML_PATH = __import__("pathlib").Path(__file__).parent / "index.html"
FRONTEND_HTML = _HTML_PATH.read_text(encoding="utf-8")

# ── FastAPI app ───────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("="*60)
    log.info("AI Sheep Analyzer — Production Hybrid Mode v2.0")
    log.info(f"  YOLO:          {'✓ YOLOv8x' if YOLO_MODEL  else '✗ pixel fallback'}")
    log.info(f"  Custom Model:  {'✓ EfficientNetV2-ONNX' if CUSTOM_MODEL else '✗ not loaded (run train_custom.py)'}")  # NEW
    log.info(f"  CLIP-1:        {'✓ ViT-L/14@336px' if CLIP_MODEL  else '✗ pixel fallback'}")
    log.info(f"  CLIP-2:        {'✓ ViT-L/14' if CLIP_MODEL_2 else '✗ not loaded'}")
    log.info(f"  BioBERT:       {'✓ loaded' if VET_MODEL  else '✗ dict fallback'}")
    log.info(f"  Device:        {DEVICE}")
    log.info(f"  Prometheus:    {'✓ enabled' if PROMETHEUS_AVAILABLE else '✗ disabled'}")  # NEW
    log.info(f"  MLflow:        {'✓ enabled' if MLFLOW_AVAILABLE else '✗ disabled'}")      # NEW
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

# ── NEW: Prometheus metrics scrape endpoint ───────────────────────
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
        "custom_model": CUSTOM_MODEL is not None,       # NEW
        "clip":         CLIP_MODEL is not None,
        "clip2":        CLIP_MODEL_2 is not None,
        "vet_model":    VET_MODEL is not None,
        "device":       DEVICE,
        "prometheus":   PROMETHEUS_AVAILABLE,           # NEW
        "mlflow":       MLFLOW_AVAILABLE,               # NEW
        "pipeline":     "YOLO → Custom → CLIP → Pixel", # NEW
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
        # NEW: track diagnose calls in Prometheus
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
        # NEW: Prometheus active request tracking
        if PROMETHEUS_AVAILABLE: ACTIVE_REQUESTS.inc()
        t0 = time.perf_counter()

        with (PREDICTION_LATENCY.time() if PROMETHEUS_AVAILABLE else _nullctx()):
            result = classify(img)

        elapsed = round((time.perf_counter()-t0)*1000)
        result.pop("_price_breed", None)

        # NEW: emit Prometheus metrics
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

        # NEW: log to MLflow if available
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
                pass  # never let MLflow crash the request

        return JSONResponse({**result, "quality": image_quality(img), "elapsed_ms": elapsed})
    except Exception as e:
        log.error(traceback.format_exc())
        raise HTTPException(500, f"Classification failed: {e}")
    finally:
        if PROMETHEUS_AVAILABLE: ACTIVE_REQUESTS.dec()

# ── NEW: tiny context manager for when Prometheus is absent ───────
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
