import argparse, datetime, io, logging, sys, time, traceback, warnings
from contextlib import asynccontextmanager, contextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

warnings.filterwarnings("ignore")

# ── Helper: Null Context for optional Prometheus ──────────────────
@contextmanager
def _nullctx():
    yield

# ── Optional deps ─────────────────────────────────────────────────
try:
    import multipart
except ImportError:
    sys.exit("\n[ERROR] pip install python-multipart\n")

try:
    import cv2; CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# ── Prometheus metrics ───────────────────────────────────────
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response as FastAPIResponse
    PROMETHEUS_AVAILABLE = True

    PREDICTIONS_TOTAL    = Counter("sheep_predictions_total", "Total predictions", ["breed", "confidence_bucket", "pipeline"])
    PREDICTION_LATENCY   = Histogram("sheep_prediction_latency_seconds", "Latency", buckets=[.05,.1,.25,.5,1,2,5])
    CONFIDENCE_HISTOGRAM = Histogram("sheep_prediction_confidence", "Confidence scores", buckets=[.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
    ACTIVE_REQUESTS      = Gauge("sheep_active_requests", "Active requests")
    DIAGNOSE_TOTAL       = Counter("sheep_diagnose_total", "Diagnose calls", ["status"])
    CUSTOM_MODEL_HITS    = Counter("sheep_custom_model_hits", "Custom model usage", ["result"])
    DATA_DRIFT_SCORE     = Gauge("sheep_data_drift_score", "Rolling avg confidence")
    _confidence_window   = []

    def _update_drift(conf: float):
        _confidence_window.append(conf)
        if len(_confidence_window) > 100: _confidence_window.pop(0)
        DATA_DRIFT_SCORE.set(sum(_confidence_window) / len(_confidence_window))
except ImportError:
    PROMETHEUS_AVAILABLE = False
    def _update_drift(conf): pass

# ── MLflow experiment tracking ──────────────────────────────
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("sheep")

# ── Constants & Metadata ──────────────────────────────────────────
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
BREED_META   = {"Lohi": ("Sindh/Punjab","Wool, Meat"), "Kajli": ("Sindh/Punjab","Meat, Breeding"), "Lari": ("Sindh (coastal)","Meat")}

DISEASE_DB = {
    "Foot Rot": {"symptoms":{"limping":0.8,"skin lesions":0.4,"weakness":0.2}, "severity":"Moderate", "actions":["Clean hoof area","Zinc sulfate footbath","Dry ground"], "description":"Bacterial hoof infection."},
    "Sheep Pox": {"symptoms":{"fever":0.7,"skin lesions":0.9,"loss of appetite":0.5}, "severity":"High", "actions":["Isolate","Antiseptic","Vaccinate"], "description":"Contagious viral disease."},
    "Pneumonia": {"symptoms":{"fever":0.6,"coughing":0.9,"nasal discharge":0.8}, "severity":"High", "actions":["Warm shelter","Antibiotics","Ventilation"], "description":"Respiratory infection."},
    "Enterotoxemia": {"symptoms":{"diarrhea":0.6,"swollen belly":0.5,"weakness":0.8}, "severity":"Critical", "actions":["Reduce grain","CD&T Vac","Vet help"], "description":"Overeating disease."},
    "Parasite Infestation":{"symptoms":{"diarrhea":0.5,"weight loss":0.9,"weakness":0.6}, "severity":"Moderate", "actions":["Deworming","Rotate pasture"], "description":"Worm infestation."},
    "Mastitis": {"symptoms":{"udder swelling":1.0,"fever":0.4}, "severity":"High", "actions":["Clean bedding","Hot compress"], "description":"Mammary inflammation."},
    "Bloat": {"symptoms":{"swollen belly":1.0,"weakness":0.4}, "severity":"Critical", "actions":["Walk sheep","Anti-bloat drench"], "description":"Gas buildup in rumen."},
    "Anthrax": {"symptoms":{"fever":0.9,"weakness":0.7}, "severity":"Critical", "actions":["Contact authorities","Do NOT open carcass"], "description":"Fatal bacterial disease."},
    "Scrapie": {"symptoms":{"weakness":0.5,"wool loss":0.6,"weight loss":0.7}, "severity":"Critical", "actions":["Report","Cull affected"], "description":"Neurological disease."},
}

SYMPTOM_EXPANSIONS = {
    "fever": "elevated body temperature hyperthermia", "coughing": "respiratory cough hacking",
    "nasal discharge": "nasal mucus secretion", "diarrhea": "loose watery feces",
    "swollen belly": "abdominal distension bloating", "loss of appetite": "anorexia inappetence",
    "weakness": "lethargy depression", "limping": "lameness difficulty walking",
    "skin lesions": "dermal lesions pustules", "wool loss": "alopecia dermatitis",
    "udder swelling": "mastitis inflammation", "weight loss": "cachexia emaciation",
}

# ── Prompts ───────────────────────────────────────────────────────
BREED_PROMPTS = {
    "Lohi":  ["Lohi sheep tan face Punjab Pakistan", "tan pinkish face thick white wool body"],
    "Kajli": ["Kajli sheep white face Roman nose", "white face dark eye rings tall long legs"],
    "Lari":  ["Lari sheep faded brown coastal face Sindh", "Jhari breed faded light brown face"],
}
AGE_PROMPTS = {
    "lamb":  ["newborn lamb sheep under 6 months"], "young": ["juvenile sheep 6 to 18 months"],
    "adult": ["mature adult sheep 1.5 to 4 years"], "old": ["old aged sheep over 4 years"],
}
HEALTH_PROMPTS = {
    "healthy": ["healthy vibrant coat alert posture"], "weak": ["weak lethargic dull coat"],
    "diseased": ["sick diseased sheep visible illness"],
}

# ── Model Initialization ──────────────────────────────────────────
YOLO_MODEL = CLIP_MODEL = CLIP_MODEL_2 = CLIP_PREPROCESS = CLIP_PREPROCESS_2 = VET_MODEL = CUSTOM_MODEL = CUSTOM_PREPROCESS = None
DEVICE = "cpu"; VET_DEVICE = -1; CUSTOM_THRESHOLD = 0.75

try:
    import torch
    if torch.cuda.is_available(): DEVICE = "cuda"; VET_DEVICE = 0
    from ultralytics import YOLO
    YOLO_MODEL = YOLO("yolov8x.pt")
    YOLO_MODEL.to(DEVICE)
    import clip
    CLIP_MODEL, CLIP_PREPROCESS = clip.load("ViT-L/14@336px", device=DEVICE)
    CLIP_MODEL_2, CLIP_PREPROCESS_2 = clip.load("ViT-L/14", device=DEVICE)
    from transformers import pipeline as hf_pipeline
    VET_MODEL = hf_pipeline("zero-shot-classification", model="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb", device=VET_DEVICE)
    import onnxruntime as ort
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if DEVICE == "cuda" else ["CPUExecutionProvider"]
    CUSTOM_MODEL = ort.InferenceSession("sheep_breed_classifier.onnx", providers=providers)
    import albumentations as A
    CUSTOM_PREPROCESS = A.Compose([A.Resize(224, 224), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    log.info(f"✓ All AI Layers loaded on {DEVICE}")
except Exception as e:
    log.warning(f"Note: Some AI models failed to load ({e}). System will use available fallbacks.")

# ── Core Logic ────────────────────────────────────────────────────
def _pixel_fallback(img):
    # Simplified version of your pixel logic for quick results or deep failure
    w, h = img.size
    face = np.array(img.crop((int(w*.25), 0, int(w*.75), int(h*.30))).convert("RGB")).mean(axis=(0,1))
    r, g, b = face
    if r > 180 and g > 180 and b > 180: breed, conf, reason = "Kajli", 0.76, "White face detected"
    elif r > 140 and r > g: breed, conf, reason = "Lohi", 0.75, "Tan face detected"
    else: breed, conf, reason = "Lari", 0.55, "Faded face detected"
    return {"is_sheep": True, "breed": breed, "breed_confidence": conf, "top_breeds": [{"breed":breed,"probability":conf}], "age": "adult", "age_confidence": 0.5, "health": "healthy", "health_confidence": 0.5, "pipeline": "Pixel Fallback", "reason": reason}

def _clip_score(model, preprocess, img_pil, prompt_dict, device):
    import torch, clip as clip_lib
    img_t = preprocess(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        img_f = model.encode_image(img_t)
        img_f /= img_f.norm(dim=-1, keepdim=True)
        return {label: sum((img_f @ (model.encode_text(clip_lib.tokenize([p]).to(device)).float().T / 
                model.encode_text(clip_lib.tokenize([p]).to(device)).float().norm())).item() for p in prompts)/len(prompts) for label, prompts in prompt_dict.items()}

def _softmax_probs(scores, temp=5.0):
    import math
    exps = {k: math.exp(v*temp) for k,v in scores.items()}
    total = sum(exps.values())
    return {k: v/total for k,v in exps.items()}

def classify(img: Image.Image) -> dict:
    if CLIP_MODEL is None: return _pixel_fallback(img)
    try:
        sheep_img, yolo_crop = img, False
        if YOLO_MODEL:
            res = YOLO_MODEL(img, verbose=False)
            boxes = [box for r in res for box in r.boxes if YOLO_MODEL.names[int(box.cls[0])] == "sheep"]
            if boxes:
                b = boxes[0].xyxy[0].tolist()
                sheep_img = img.crop((max(0, b[0]), max(0, b[1]), min(img.width, b[2]), min(img.height, b[3])))
                yolo_crop = True
        
        # Layer 2: Custom ONNX
        if CUSTOM_MODEL:
            img_np = np.array(sheep_img.convert("RGB").resize((224, 224)), dtype=np.float32)
            arr = np.transpose(CUSTOM_PREPROCESS(image=img_np)["image"], (2, 0, 1))[np.newaxis, ...]
            raw = CUSTOM_MODEL.run(None, {CUSTOM_MODEL.get_inputs()[0].name: arr})[0][0]
            probs = np.exp(raw) / np.sum(np.exp(raw))
            if np.max(probs) >= CUSTOM_THRESHOLD:
                breed = BREEDS[np.argmax(probs)]
                return _validate_classify({"is_sheep":True, "breed":breed, "breed_confidence":round(float(np.max(probs)),3), "top_breeds":[{"breed":BREEDS[i],"probability":round(float(p),3)} for i,p in enumerate(probs)], "age":"adult", "age_confidence":0.6, "health":"healthy", "health_confidence":0.6, "yolo_crop":yolo_crop, "pipeline":"Custom ONNX", "reason":"High-confidence fine-tuned hit"})

        # Layer 3: CLIP Ensemble
        s1 = _clip_score(CLIP_MODEL, CLIP_PREPROCESS, sheep_img, BREED_PROMPTS, DEVICE)
        probs = _softmax_probs(s1)
        breed = max(probs, key=probs.get)
        
        age_scores = _clip_score(CLIP_MODEL, CLIP_PREPROCESS, sheep_img, AGE_PROMPTS, DEVICE)
        health_scores = _clip_score(CLIP_MODEL, CLIP_PREPROCESS, sheep_img, HEALTH_PROMPTS, DEVICE)

        return _validate_classify({
            "is_sheep": True, "breed": breed, "breed_confidence": round(probs[breed],3),
            "top_breeds": [{"breed":b,"probability":round(p,3)} for b,p in sorted(probs.items(), key=lambda x:x[1], reverse=True)],
            "age": max(age_scores, key=age_scores.get), "age_confidence": 0.8,
            "health": max(health_scores, key=health_scores.get), "health_confidence": 0.8,
            "yolo_crop": yolo_crop, "bcs": 3, "wool": "Fine" if breed=="Lohi" else "Medium", "use": BREED_META[breed][1],
            "pipeline": "CLIP Ensemble", "reason": "ViT-L/14 detection", "_price_breed": breed
        })
    except Exception as e:
        log.error(f"Classify failed: {e}")
        return _pixel_fallback(img)

def diagnose(selected: list) -> dict:
    if not selected: return {"predicted_diseases":[], "health_score":100, "status":"Healthy", "diet":[]}
    if VET_MODEL:
        clinical = f"Ovine patient: {', '.join(SYMPTOM_EXPANSIONS.get(s,s) for s in selected)}."
        ai_res = VET_MODEL(clinical, list(DISEASE_DB.keys()), multi_label=True)
        preds = [{"disease": d, "probability": round(s,2), "severity": DISEASE_DB[d]["severity"], "actions": DISEASE_DB[d]["actions"]} for d, s in zip(ai_res["labels"], ai_res["scores"]) if s > 0.2]
        return {"predicted_diseases": preds, "health_score": 100 - int(len(selected)*10), "status": "Risk" if preds else "Healthy", "diet": ["Fodder", "Water"]}
    return {"predicted_diseases":[], "health_score":80, "status":"Manual Check Required", "diet":[]}

# ── Price & Market ────────────────────────────────────────────────
def price(breed, age, health, wkg):
    is_eid = datetime.datetime.now().strftime("%B") in EID_MONTHS
    rate = BREED_RATE.get(breed, 600) * AGE_FACTOR.get(age, 1) * HEALTH_FACTOR.get(health, 1) * (EID_FACTOR if is_eid else 1)
    est = int(wkg * rate)
    return {"weight_kg":wkg, "estimated":est, "display":f"PKR {est:,}", "formula":f"{wkg}kg x {int(rate)} PKR/kg"}

def _validate_classify(r):
    r.setdefault("bcs", 3); r.setdefault("wool", "Medium"); r.setdefault("use", "Meat")
    return r

# ── FastAPI App ───────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Sheep AI Pipeline Online: YOLOv8x -> Custom -> CLIP -> Pixel")
    if MLFLOW_AVAILABLE: mlflow.set_experiment("sheep-classifier")
    yield

app = FastAPI(title="AI Sheep Analyzer Pro", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/metrics")
def metrics():
    if not PROMETHEUS_AVAILABLE: raise HTTPException(503, "Prometheus disabled")
    return FastAPIResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
def health_check():
    return {"status": "ok", "gpu": DEVICE == "cuda", "layers": ["YOLOv8x", "ONNX", "CLIP-L14", "BioBERT"]}

@app.post("/classify")
async def do_classify(file: UploadFile = File(...)):
    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    if PROMETHEUS_AVAILABLE: ACTIVE_REQUESTS.inc()
    t0 = time.perf_counter()
    
    try:
        with (PREDICTION_LATENCY.time() if PROMETHEUS_AVAILABLE else _nullctx()):
            result = classify(img)
        
        elapsed = int((time.perf_counter() - t0) * 1000)
        
        if MLFLOW_AVAILABLE and result.get("is_sheep"):
            with mlflow.start_run(nested=True):
                mlflow.log_param("breed", result["breed"])
                mlflow.log_metric("conf", result["breed_confidence"])
        
        return JSONResponse({**result, "elapsed_ms": elapsed})
    finally:
        if PROMETHEUS_AVAILABLE: ACTIVE_REQUESTS.dec()

@app.post("/diagnose")
async def do_diagnose(data: dict):
    return JSONResponse(diagnose(data.get("symptoms", [])))

@app.post("/price")
async def calc_price(data: dict):
    return JSONResponse(price(data.get("breed","Lohi"), data.get("age","adult"), data.get("health","healthy"), float(data.get("weight_kg", 40))))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5050)
