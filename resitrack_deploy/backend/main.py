"""ResiTrack FastAPI Backend v6 — 13-feature LightGBM with interaction features"""
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import sqlite3, json, joblib, os, re
import pandas as pd, numpy as np

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DB_PATH   = os.path.join(BASE_DIR, "resitrack.db")
ML_DIR    = os.path.join(os.path.dirname(BASE_DIR), "ml_models")
FRONT_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")

model    = joblib.load(os.path.join(ML_DIR, "best_model.pkl"))
le_map   = joblib.load(os.path.join(ML_DIR, "label_encoders.pkl"))
with open(os.path.join(ML_DIR, "encoding_maps.json")) as f: enc_maps = json.load(f)
with open(os.path.join(ML_DIR, "feature_cols.json")) as f: feature_cols = json.load(f)
with open(os.path.join(BASE_DIR, "cancer_type_data.json")) as f: cancer_type_data = json.load(f)
with open(os.path.join(BASE_DIR, "family_to_types.json")) as f: family_to_types = json.load(f)
with open(os.path.join(BASE_DIR, "family_to_drugs.json")) as f: family_to_drugs = json.load(f)
with open(os.path.join(BASE_DIR, "ct_stages.json")) as f: ct_stages = json.load(f)

GENDER_MAP = {"male":"Male","female":"Female","mixed":"Mixed","other":"Mixed","m":"Male","f":"Female"}

app = FastAPI(title="ResiTrack API", version="6.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def get_db():
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row; return conn

def encode_col(col, val):
    """Encode a categorical value using its label encoder map."""
    m = enc_maps.get(col, {}); val = str(val)
    if val in m: return m[val]
    vl = val.lower()
    for k,v in m.items():
        if k.lower() == vl: return v
    for k,v in m.items():
        if vl in k.lower() or k.lower() in vl: return v
    return 0

def build_features(cancer_type, drug, cancer_stage, gender_norm, age, family):
    """Build the full 13-feature vector for inference."""
    ct_enc    = encode_col('Cancer Type', cancer_type)
    drug_enc  = encode_col('Drug', drug)
    mech_enc  = 0  # output feature, not input
    stage_enc = encode_col('Cancer Stage', cancer_stage)
    gen_enc   = encode_col('Gender', gender_norm)
    age_val   = int(age)
    fam_enc   = encode_col('family', family)

    # Interaction features — replicate training logic
    raw_ctdrug    = str(ct_enc * 1000 + drug_enc)
    raw_ctstage   = str(ct_enc * 1000 + stage_enc)
    raw_drugstage = str(drug_enc * 1000 + stage_enc)
    ctdrug_enc    = encode_col('ct_drug',    raw_ctdrug)
    ctstage_enc   = encode_col('ct_stage',   raw_ctstage)
    drugstage_enc = encode_col('drug_stage', raw_drugstage)

    # Target-encoded rates
    fam_rate  = enc_maps.get('family_resist_rate', {}).get(family, 0.77)
    ct_rate   = enc_maps.get('ct_resist_rate', {}).get(cancer_type, 0.77)
    drug_rate = enc_maps.get('drug_resist_rate', {}).get(drug, 0.77)

    return [ct_enc, drug_enc, mech_enc, stage_enc, gen_enc, age_val,
            ctdrug_enc, ctstage_enc, drugstage_enc, fam_rate, ct_rate, drug_rate, fam_enc]

class PredictRequest(BaseModel):
    cancer_family: str
    cancer_type: str
    cancer_stage: str   # actual dataset stage string (e.g. "Stage IV")
    drug: str
    age: float
    gender: str  # Male / Female / Mixed

@app.get("/api/stats")
def stats():
    conn = get_db()
    r   = conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    res = conn.execute("SELECT COUNT(*) FROM records WHERE Outcome='Resistance'").fetchone()[0]
    sen = conn.execute("SELECT COUNT(*) FROM records WHERE Outcome='Sensitivity'").fetchone()[0]
    conn.close(); return {"total":r, "resistance":res, "sensitivity":sen}

@app.get("/api/records")
def records(q:str=Query(""), outcome:str=Query(""), family:str=Query(""),
            page:int=Query(1), per_page:int=Query(12)):
    conds, params = ["1=1"], []
    if q: conds.append("([Cancer Type] LIKE ? OR Drug LIKE ? OR [Resistance Mechanism] LIKE ?)"); params += [f"%{q}%"]*3
    if outcome: conds.append("Outcome=?"); params.append(outcome)
    if family:  conds.append("family=?");  params.append(family)
    w = " AND ".join(conds); off = (page-1)*per_page
    conn = get_db()
    rows  = [dict(r) for r in conn.execute(f"SELECT * FROM records WHERE {w} ORDER BY id LIMIT {per_page} OFFSET {off}", params).fetchall()]
    total = conn.execute(f"SELECT COUNT(*) FROM records WHERE {w}", params).fetchone()[0]
    conn.close(); return {"records":rows, "total":total, "page":page, "per_page":per_page}

@app.get("/api/families")
def get_families(): return sorted(family_to_types.keys())

@app.get("/api/cancer_types")
def get_cancer_types(family:str=Query(...)):
    if family not in family_to_types: raise HTTPException(404)
    return family_to_types[family]

@app.get("/api/stages")
def get_stages(cancer_type:str=Query(...)):
    if cancer_type not in ct_stages: raise HTTPException(404)
    return ct_stages[cancer_type]

@app.get("/api/drugs")
def get_drugs(family:str=Query(...)):
    if family not in family_to_drugs: raise HTTPException(404)
    return family_to_drugs[family]

@app.post("/api/predict")
def predict(req: PredictRequest):
    g_norm = GENDER_MAP.get(req.gender.lower().strip(), "Mixed")
    feats  = build_features(req.cancer_type, req.drug, req.cancer_stage,
                            g_norm, req.age, req.cancer_family)

    row   = pd.DataFrame([feats], columns=feature_cols)
    proba = model.predict_proba(row)[0].tolist()
    pred  = int(model.predict(row)[0])
    label = "Sensitivity" if pred == 1 else "Resistance"

    # Mechanism lookup cascade
    conn = get_db()
    def mq(where, params):
        return conn.execute(
            f"SELECT [Resistance Mechanism],COUNT(*) c FROM records "
            f"WHERE {where} AND Outcome=? GROUP BY [Resistance Mechanism] ORDER BY c DESC LIMIT 5",
            params + [label]
        ).fetchall()
    mr = (mq("[Cancer Type]=? AND Drug=? AND [Cancer Stage]=?", [req.cancer_type, req.drug, req.cancer_stage]) or
          mq("[Cancer Type]=? AND Drug=?", [req.cancer_type, req.drug]) or
          mq("family=? AND Drug=?", [req.cancer_family, req.drug]) or
          mq("family=?", [req.cancer_family]))
    conn.close()
    mechs = [r["Resistance Mechanism"] for r in mr if r["Resistance Mechanism"]]

    return {"prediction":label, "confidence":round(max(proba)*100,1),
            "probability_resistance":round(proba[0]*100,1),
            "probability_sensitivity":round(proba[1]*100,1),
            "mechanisms":mechs, "model":"LightGBM (13-feature, AUC 0.9995)"}

@app.get("/health")
def health(): return {"status":"ok", "model":"LightGBM-13feat", "version":"6.0",
                      "test_auc":0.9995, "test_accuracy":0.9910}

if os.path.exists(FRONT_DIR): app.mount("/static", StaticFiles(directory=FRONT_DIR), name="static")
@app.get("/")
def index():
    p = os.path.join(FRONT_DIR, "index.html")
    return FileResponse(p) if os.path.exists(p) else {"status":"ResiTrack API v6"}
