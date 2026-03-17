
import argparse
import datetime
import logging
import sys
import warnings
from contextlib import asynccontextmanager

warnings.filterwarnings("ignore")

try:
    import multipart  # noqa: F401
except ImportError:
    sys.exit("[ERROR] pip install python-multipart")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sheep.light")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BREEDS        = ["Lohi", "Kajli", "Lari"]
EID_MONTHS    = {"July", "August", "September", "October"}
VALID_BREEDS  = {"Lohi", "Kajli", "Lari"}
VALID_AGES    = {"lamb", "young", "adult", "old"}
VALID_HEALTH  = {"healthy", "weak", "diseased"}

BREED_RATE    = {"Lohi": 650,  "Kajli": 700, "Lari": 600}
AGE_FACTOR    = {"lamb": 1.20, "young": 1.10, "adult": 1.00, "old": 0.75}
HEALTH_FACTOR = {"healthy": 1.00, "weak": 0.60, "diseased": 0.30}
EID_FACTOR    = 1.60

BREED_META = {
    "Lohi":  ("Sindh/Punjab",    "Wool, Meat"),
    "Kajli": ("Sindh/Punjab",    "Meat, Breeding"),
    "Lari":  ("Sindh (coastal)", "Meat"),
}

DISEASE_DB = {
    "Foot Rot": {
        "symptoms":    {"limping": 0.8, "skin lesions": 0.4, "weakness": 0.2},
        "severity":    "Moderate",
        "description": "Bacterial hoof infection causing painful lameness.",
        "actions":     ["Clean hoof area", "Zinc sulfate footbath 2-3x/week",
                        "Keep on dry ground", "Isolate infected animals", "Consult vet if worsening"],
    },
    "Sheep Pox": {
        "symptoms":    {"fever": 0.7, "skin lesions": 0.9, "loss of appetite": 0.5, "nasal discharge": 0.3},
        "severity":    "High",
        "description": "Highly contagious viral skin disease. Progresses to pustules over 2 weeks.",
        "actions":     ["Isolate immediately", "Apply antiseptic to lesions",
                        "Vaccinate healthy flock", "Contact vet for antiviral treatment"],
    },
    "Pneumonia": {
        "symptoms":    {"fever": 0.6, "coughing": 0.9, "nasal discharge": 0.8, "weakness": 0.4},
        "severity":    "High",
        "description": "Respiratory infection from stress or dust.",
        "actions":     ["Provide warm shelter", "Antibiotic treatment (Vet)",
                        "Improve ventilation", "Reduce overcrowding"],
    },
    "Enterotoxemia": {
        "symptoms":    {"diarrhea": 0.6, "swollen belly": 0.5, "weakness": 0.8, "loss of appetite": 0.4},
        "severity":    "Critical",
        "description": "Overeating disease caused by Clostridium. Rapidly fatal if untreated.",
        "actions":     ["Reduce grain intake immediately", "CD&T Vaccination",
                        "Immediate Vet attention", "Provide oral electrolytes"],
    },
    "Parasite Infestation": {
        "symptoms":    {"diarrhea": 0.5, "weight loss": 0.9, "wool loss": 0.4, "weakness": 0.6},
        "severity":    "Moderate",
        "description": "Worm infestation. Manage with deworming.",
        "actions":     ["Deworming schedule", "Rotate pastures",
                        "Check eye membranes (FAMACHA)", "Fecal egg count test"],
    },
    "Mastitis": {
        "symptoms":    {"udder swelling": 1.0, "fever": 0.4, "loss of appetite": 0.3},
        "severity":    "High",
        "description": "Mammary gland inflammation, often bacterial.",
        "actions":     ["Keep bedding clean/dry", "Hot compresses on udder",
                        "Intramammary antibiotics", "Strip affected quarters frequently"],
    },
    "Bloat": {
        "symptoms":    {"swollen belly": 1.0, "loss of appetite": 0.6, "weakness": 0.4},
        "severity":    "Critical",
        "description": "Gas buildup in rumen. Life-threatening if untreated quickly.",
        "actions":     ["Walk sheep slowly", "Anti-bloat drench",
                        "Avoid wet lush clover/alfalfa", "Stomach tube if severe"],
    },
    "Anthrax": {
        "symptoms":    {"fever": 0.9, "weakness": 0.7, "loss of appetite": 0.5},
        "severity":    "Critical",
        "description": "Rare but fatal bacterial disease. Sudden death common. Report immediately.",
        "actions":     ["Contact authorities immediately", "Do NOT open carcass",
                        "Vaccinate entire flock", "Quarantine affected area"],
    },
    "Scrapie": {
        "symptoms":    {"weakness": 0.5, "wool loss": 0.6, "weight loss": 0.7},
        "severity":    "Critical",
        "description": "Progressive neurological disease. No treatment available.",
        "actions":     ["Report to animal health authorities", "Cull affected animals",
                        "Genetic testing for flock", "Do not breed from affected lines"],
    },
}

# ---------------------------------------------------------------------------
# Business logic
# ---------------------------------------------------------------------------

def _diet_advice(symptoms: list) -> list:
    if any(s in symptoms for s in ("diarrhea", "swollen belly")):
        return ["Dry hay only", "Reduce grain intake", "Increase hydration", "Electrolyte solution"]
    if any(s in symptoms for s in ("weakness", "weight loss")):
        return ["Protein supplements", "Cottonseed cake", "Fresh grass", "Clean water", "Vitamin B complex"]
    return ["Green fodder", "Wheat straw", "Corn feed", "Mineral salt block", "Fresh clean water"]


def _calc_price(breed: str, age: str, health: str, weight_kg: float) -> dict:
    is_eid = datetime.datetime.now().strftime("%B") in EID_MONTHS
    base   = BREED_RATE.get(breed, BREED_RATE["Lohi"])
    af     = AGE_FACTOR.get(age, 1.0)
    hf     = HEALTH_FACTOR.get(health, 1.0)
    ef     = EID_FACTOR if is_eid else 1.0
    rpk    = base * af * hf * ef
    est    = int(weight_kg * rpk)
    return {
        "weight_kg":     round(weight_kg, 1),
        "base_rate":     base,
        "age_factor":    af,
        "health_factor": hf,
        "eid_factor":    round(ef, 2),
        "rate_per_kg":   round(rpk),
        "estimated":     est,
        "min":           int(est * 0.92),
        "max":           int(est * 1.08),
        "eid":           is_eid,
        "display":       f"PKR {int(est * 0.92):,} – {int(est * 1.08):,}",
        "formula":       f"{weight_kg:.1f} kg × {round(rpk):,} PKR/kg",
    }


def _calc_market_prices() -> dict:
    is_eid = datetime.datetime.now().strftime("%B") in EID_MONTHS
    rows   = []
    for breed, base in BREED_RATE.items():
        row = {
            "breed":        breed,
            "eid":          is_eid,
            "region":       BREED_META[breed][0],
            "uses":         BREED_META[breed][1],
            "base_rate_kg": base,
        }
        for age, af in AGE_FACTOR.items():
            for hlth, hf in HEALTH_FACTOR.items():
                ef  = EID_FACTOR if is_eid else 1.0
                r   = round(base * af * hf * ef)
                ref = int(40 * r)
                row[f"{age}_{hlth}_rate"] = r
                row[f"{age}_{hlth}_min"]  = int(ref * 0.92)
                row[f"{age}_{hlth}_max"]  = int(ref * 1.08)
        rows.append(row)
    return {
        "prices":     rows,
        "eid":        is_eid,
        "ref_weight": 40,
        "updated":    datetime.datetime.now().isoformat(),
    }


def _diagnose(symptoms: list) -> dict:
    if not symptoms:
        return {"predicted_diseases": [], "health_score": 100,
                "status": "Healthy", "diet": _diet_advice(symptoms)}

    preds = []
    for disease, data in DISEASE_DB.items():
        matched = {s: data["symptoms"][s] for s in symptoms if s in data["symptoms"]}
        if not matched:
            continue
        score = min(0.98,
            (sum(matched.values()) / sum(data["symptoms"].values()))
            * (len(matched) / len(data["symptoms"]))
        )
        preds.append({
            "disease":     disease,
            "probability": round(score, 2),
            "severity":    data["severity"],
            "actions":     data["actions"],
            "description": data["description"],
        })

    preds.sort(key=lambda x: x["probability"], reverse=True)
    health_score = max(0, min(100, 100 - len(symptoms) * 12))
    status = "High Risk" if health_score < 40 else "Moderate Risk" if health_score < 75 else "Healthy"

    return {
        "predicted_diseases": preds,
        "health_score":       health_score,
        "status":             status,
        "diet":               _diet_advice(symptoms),
    }


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("AI Sheep Analyzer — Light Backend ready")
    yield


app = FastAPI(title="AI Sheep Analyzer — Light Backend", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "backend": "light",
            "classify": "Hugging Face Spaces"}


@app.get("/breeds")
def breeds():
    return {"breeds": [{"name": b, "base_rate_kg": BREED_RATE[b]} for b in BREEDS]}


@app.get("/market-prices")
def market_prices():
    return JSONResponse(_calc_market_prices())


@app.post("/price")
async def calc_price(data: dict):
    try:
        w = float(data.get("weight_kg", 0))
    except (TypeError, ValueError):
        w = 0.0
    if w <= 0:
        raise HTTPException(400, "weight_kg must be a positive number")
    return JSONResponse(_calc_price(
        str(data.get("breed",  "Lohi")),
        str(data.get("age",    "adult")),
        str(data.get("health", "healthy")),
        w,
    ))


@app.post("/diagnose")
async def diagnose(data: dict):
    symptoms = data.get("symptoms", [])
    if not isinstance(symptoms, list):
        raise HTTPException(400, "symptoms must be a list")
    return JSONResponse(_diagnose(symptoms))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port, reload=False)
