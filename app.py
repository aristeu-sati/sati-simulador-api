from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from math import pow
from datetime import date
from typing import Optional, Dict, Any, List
import time
import os
import re

import pandas as pd

app = FastAPI(title="Sati Simulador API")

# =========================================================
# CONFIG (Google Sheets CSV + TTL cache)
# =========================================================

FIN_URL_DEFAULT = "https://docs.google.com/spreadsheets/d/1ONyZQvlJTIYSmHX4fGZTtH5j3N5yalMjrGMfPsyrEbE/gviz/tq?tqx=out:csv&gid=0"
INS_URL_DEFAULT = "https://docs.google.com/spreadsheets/d/1ONyZQvlJTIYSmHX4fGZTtH5j3N5yalMjrGMfPsyrEbE/gviz/tq?tqx=out:csv&gid=1836407575"

FIN_URL = os.getenv("FIN_URL", FIN_URL_DEFAULT)
INS_URL = os.getenv("INS_URL", INS_URL_DEFAULT)
CONFIG_TTL_SECONDS = int(os.getenv("CONFIG_TTL_SECONDS", "300"))  # 5 minutos

_fin_df: Optional[pd.DataFrame] = None
_ins_df: Optional[pd.DataFrame] = None
_last_load_ts: float = 0.0

def _norm_col(c: str) -> str:
    # normaliza cabeçalho: tira espaços, lower
    return str(c).strip().lower()

def _to_bool(x) -> bool:
    """Normaliza valores tipo TRUE/FALSE, Verdadeiro/Falso, 1/0, Sim/Não."""
    if x is None:
        return False
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in {"true", "verdadeiro", "1", "sim", "yes", "y"}

def _to_float(x):
    """Converte número vindo como '0.123' ou '0,123'."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "":
        return None
    # tenta lidar com formatos 1.234,56 e 1234,56
    if s.count(",") == 1 and s.count(".") >= 1:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except:
        return None

def _extract_age_from_label(label: Any) -> Optional[int]:
    # "MIP por idade - 35" -> 35
    if label is None:
        return None
    m = re.search(r"(\d+)\s*$", str(label).strip())
    return int(m.group(1)) if m else None

def load_configs(force: bool = False) -> None:
    """Carrega as configs do Google Sheets. Usa cache por TTL."""
    global _fin_df, _ins_df, _last_load_ts
    now = time.time()

    if not force and _fin_df is not None and _ins_df is not None:
        if (now - _last_load_ts) < CONFIG_TTL_SECONDS:
            return

    try:
        fin_raw = pd.read_csv(FIN_URL)
        ins_raw = pd.read_csv(INS_URL)
    except Exception as e:
        raise RuntimeError(f"Falha ao carregar configs do Google Sheets CSV: {e}")

    # Normaliza nomes das colunas (lower + strip), mas mantém um mapa para acessar
    fin = fin_raw.copy()
    fin.columns = [_norm_col(c) for c in fin.columns]

    ins = ins_raw.copy()
    ins.columns = [_norm_col(c) for c in ins.columns]

    # ---------------------------
    # FINANCIAMENTO (colunas mínimas)
    # ---------------------------
    # Aceita variações comuns: "amortização" pode vir com ou sem acento etc.
    # Aqui assumimos que sua aba está igual a que você já usa.
    required_fin = [
        "banco",
        "operação",
        "amortização",
        "quota",
        "prazo máximo (meses)",
        "comprometimento de renda",
        "taxa efetiva (a.a.)",
    ]
    for c in required_fin:
        if c not in fin.columns:
            raise RuntimeError(f"Coluna obrigatória faltando em Financiamento: {c}")

    # Ativo pode ser "ativo"
    if "ativo" in fin.columns:
        fin = fin[fin["ativo"].apply(_to_bool)].copy()

    fin["banco"] = fin["banco"].astype(str).str.strip()
    fin["operação"] = fin["operação"].astype(str).str.strip()
    fin["amortização"] = fin["amortização"].astype(str).str.strip().str.upper()

    for c in ["quota", "prazo máximo (meses)", "comprometimento de renda", "taxa efetiva (a.a.)"]:
        fin[c] = fin[c].apply(_to_float)

    fin = fin.dropna(subset=[
        "banco", "operação", "amortização", "quota",
        "prazo máximo (meses)", "comprometimento de renda", "taxa efetiva (a.a.)"
    ]).copy()

    # ---------------------------
    # SEGUROS_EXPORT (aceita seu formato atual)
    # ---------------------------
    # Seu CSV tem: ativo, banco, seguradora, dfi_rate, mip_label, mip_rate, idade
    # Aqui aceitamos:
    # - banco (obrigatório)
    # - dfi_rate (obrigatório)
    # - mip_rate (obrigatório)
    # - idade (obrigatório, mas se não existir, tentamos extrair do mip_label)
    if "banco" not in ins.columns and "Banco".lower() not in ins.columns:
        # já normalizamos; então checamos "banco"
        raise RuntimeError("Coluna obrigatória faltando em Seguros_export: Banco")

    # Ativo pode ser "ativo"
    if "ativo" in ins.columns:
        ins = ins[ins["ativo"].apply(_to_bool)].copy()

    # Garante as colunas numéricas
    if "dfi_rate" not in ins.columns:
        raise RuntimeError("Coluna obrigatória faltando em Seguros_export: dfi_rate")
    if "mip_rate" not in ins.columns:
        raise RuntimeError("Coluna obrigatória faltando em Seguros_export: mip_rate")

    # idade: se não existir, tenta extrair de mip_label
    if "idade" not in ins.columns:
        if "mip_label" in ins.columns:
            ins["idade"] = ins["mip_label"].apply(_extract_age_from_label)
        else:
            raise RuntimeError("Coluna obrigatória faltando em Seguros_export: idade (ou mip_label para extrair)")

    ins["banco"] = ins["banco"].astype(str).str.strip()
    ins["idade"] = ins["idade"].apply(lambda v: int(float(v)) if str(v).strip() != "" else None)
    ins["mip_rate"] = ins["mip_rate"].apply(_to_float)
    ins["dfi_rate"] = ins["dfi_rate"].apply(_to_float)

    ins = ins.dropna(subset=["banco", "idade", "mip_rate", "dfi_rate"]).copy()

    _fin_df = fin
    _ins_df = ins
    _last_load_ts = now


@app.on_event("startup")
def _startup():
    load_configs(force=True)

# =========================================================
# Utils
# =========================================================

def annual_to_monthly(annual_rate: float) -> float:
    return pow(1.0 + annual_rate, 1.0 / 12.0) - 1.0

def calc_age(birth_date_iso: str) -> int:
    y, m, d = map(int, birth_date_iso.split("-"))
    today = date.today()
    age = today.year - y
    if (today.month, today.day) < (m, d):
        age -= 1
    return age

def clamp_age(age: int, min_age: int = 18, max_age: int = 80) -> int:
    return max(min_age, min(max_age, age))

def price_factor(i_m: float, n: int) -> float:
    if n <= 0:
        raise ValueError("n must be > 0")
    if i_m <= 0:
        return 1.0 / n
    a = pow(1.0 + i_m, n)
    return (i_m * a) / (a - 1.0)

_money_re = re.compile(r"R\$\s*([\d\.\,]+)")

def parse_brl_money(s: Any) -> Optional[float]:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    text = str(s)
    m = _money_re.search(text)
    if not m:
        return None
    raw = m.group(1).replace(".", "").replace(",", ".")
    try:
        return float(raw)
    except:
        return None

def is_min_installment_field(s: Any) -> bool:
    if s is None:
        return False
    return "parcela" in str(s).lower()

def get_insurance_rates(bank: str, age: int) -> Dict[str, float]:
    load_configs()
    if _ins_df is None:
        raise HTTPException(status_code=500, detail="Configs de seguros não carregadas")

    age_c = clamp_age(age)
    df = _ins_df[_ins_df["banco"].str.lower() == bank.strip().lower()]
    if df.empty:
        raise HTTPException(status_code=422, detail=f"Banco sem seguros_export: {bank}")

    row = df[df["idade"] == age_c]
    if row.empty:
        df2 = df.sort_values("idade")
        candidates = df2[df2["idade"] <= age_c]
        row = candidates.iloc[[-1]] if not candidates.empty else df2.iloc[[0]]

    r = row.iloc[0]
    return {"dfi_rate": float(r["dfi_rate"]), "mip_rate": float(r["mip_rate"])}

# =========================================================
# Endpoints básicos
# =========================================================

@app.get("/health")
def health():
    return {"ok": True, "ttl_seconds": CONFIG_TTL_SECONDS}

@app.get("/configs/operations")
def list_operations() -> Dict[str, Any]:
    load_configs()
    ops = sorted(_fin_df["operação"].dropna().unique().tolist())
    return {"ok": True, "operations": ops}

@app.get("/configs/banks")
def list_banks() -> Dict[str, Any]:
    load_configs()
    banks = sorted(_fin_df["banco"].dropna().unique().tolist())
    return {"ok": True, "banks": banks}

# =========================================================
# SAC (mantive os endpoints)
# =========================================================

class SacRequest(BaseModel):
    property_value: float
    quota: float
    term_months: int
    annual_rate: float
    birth_date: str
    mip_rate: float
    dfi_rate: float
    monthly_income: Optional[float] = None
    commitment_rate: Optional[float] = None

@app.post("/simulate/sac")
def simulate_sac(req: SacRequest) -> Dict[str, Any]:
    pv = req.property_value * req.quota
    down_payment = req.property_value - pv

    i_m = annual_to_monthly(req.annual_rate)
    amort = pv / req.term_months
    interest = pv * i_m

    age = calc_age(req.birth_date)
    mip = pv * req.mip_rate
    dfi = req.property_value * req.dfi_rate

    installment = amort + interest + mip + dfi

    fits = None
    max_installment = None
    if req.monthly_income and req.commitment_rate:
        max_installment = req.monthly_income * req.commitment_rate
        fits = installment <= max_installment

    return {
        "age": age,
        "monthly_rate": round(i_m, 8),
        "pv": round(pv, 2),
        "down_payment": round(down_payment, 2),
        "month1": {
            "amortization": round(amort, 2),
            "interest": round(interest, 2),
            "mip": round(mip, 2),
            "dfi": round(dfi, 2),
            "installment_total": round(installment, 2),
        },
        "income_check": {
            "max_installment": round(max_installment, 2) if max_installment is not None else None,
            "fits": fits,
        }
    }

class SacFitRequest(BaseModel):
    property_value: float
    max_quota: float
    term_months: int
    annual_rate: float
    mip_rate: float
    dfi_rate: float
    monthly_income: float
    commitment_rate: float

@app.post("/simulate/sac/fit")
def fit_sac(req: SacFitRequest) -> Dict[str, Any]:
    max_installment = req.monthly_income * req.commitment_rate
    i_m = annual_to_monthly(req.annual_rate)

    B = req.property_value * req.dfi_rate
    A = (1.0 / req.term_months) + i_m + req.mip_rate

    if max_installment <= B:
        return {
            "ok": True,
            "fits": False,
            "reason": "max_installment <= fixed_costs (DFI)",
            "max_installment_allowed": round(max_installment, 2),
            "fixed_costs_dfi": round(B, 2),
            "pv_by_income": 0.0,
            "pv_by_ltv": round(req.property_value * req.max_quota, 2),
            "pv_allowed": 0.0,
        }

    pv_by_income = (max_installment - B) / A
    pv_by_ltv = req.property_value * req.max_quota
    pv_allowed = max(0.0, min(pv_by_income, pv_by_ltv))

    return {
        "ok": True,
        "fits": pv_allowed > 0,
        "max_installment_allowed": round(max_installment, 2),
        "pv_by_income": round(pv_by_income, 2),
        "pv_by_ltv": round(pv_by_ltv, 2),
        "pv_allowed": round(pv_allowed, 2),
    }

# =========================================================
# POTENCIAL
# =========================================================

class PotentialRequest(BaseModel):
    monthly_income: float = Field(..., gt=0)
    birth_date: str = Field(..., description="YYYY-MM-DD")
    operation: str
    banks: Optional[List[str]] = None

def simulate_potential_row(
    amortization: str,  # "SAC" ou "PRICE"
    quota: float,
    term_months: int,
    annual_rate: float,
    commitment_rate: float,
    income: float,
    dfi_rate: float,
    mip_rate: float,
    min_value_field: Any,
) -> Dict[str, Any]:
    i_m = annual_to_monthly(annual_rate)
    max_installment = income * commitment_rate

    dfi_per_pv = dfi_rate / quota

    if amortization == "SAC":
        A = (1.0 / term_months) + i_m + mip_rate + dfi_per_pv
        pv_allowed = max_installment / A

        amort = pv_allowed / term_months
        interest = pv_allowed * i_m
        mip = pv_allowed * mip_rate
        dfi = (pv_allowed / quota) * dfi_rate
        installment = amort + interest + mip + dfi

    elif amortization == "PRICE":
        k = price_factor(i_m, term_months)
        A = k + mip_rate + dfi_per_pv
        pv_allowed = max_installment / A

        payment_no_insurance = pv_allowed * k
        interest = pv_allowed * i_m
        amort = payment_no_insurance - interest
        mip = pv_allowed * mip_rate
        dfi = (pv_allowed / quota) * dfi_rate
        installment = payment_no_insurance + mip + dfi

    else:
        raise ValueError(f"Unsupported amortization: {amortization}")

    property_value = pv_allowed / quota
    down_payment = property_value - pv_allowed

    min_ok = True
    min_money = parse_brl_money(min_value_field)
    if min_money is not None:
        if is_min_installment_field(min_value_field):
            if installment < min_money:
                min_ok = False
        else:
            if property_value < min_money:
                min_ok = False

    return {
        "ok": True,
        "fits": pv_allowed > 0 and min_ok,
        "quota": round(quota, 4),
        "term_months": int(term_months),
        "annual_rate": float(annual_rate),
        "commitment_rate": round(float(commitment_rate), 4),
        "max_installment_allowed": round(max_installment, 2),
        "property_value": round(property_value, 2),
        "pv_financing": round(pv_allowed, 2),
        "down_payment": round(down_payment, 2),
        "month1_installment": round(installment, 2),
        "month1_components": {
            "amortization": round(amort, 2),
            "interest": round(interest, 2),
            "mip": round(mip, 2),
            "dfi": round(dfi, 2),
        }
    }

@app.post("/simulate/potential")
def simulate_potential(req: PotentialRequest) -> Dict[str, Any]:
    load_configs()

    age = calc_age(req.birth_date)

    fin = _fin_df[_fin_df["operação"].str.lower() == req.operation.strip().lower()].copy()
    if fin.empty:
        raise HTTPException(status_code=422, detail=f"Nenhuma config para operation='{req.operation}'")

    if req.banks:
        wanted = {b.strip().lower() for b in req.banks}
        fin = fin[fin["banco"].str.lower().isin(wanted)]
        if fin.empty:
            raise HTTPException(status_code=422, detail="Filtro de banks removeu todas as configs")

    results_by_bank: Dict[str, Dict[str, Any]] = {}

    for _, row in fin.iterrows():
        bank = str(row["banco"]).strip()
        amortization = str(row["amortização"]).strip().upper()
        quota = float(row["quota"])
        term = int(float(row["prazo máximo (meses)"]))
        annual_rate = float(row["taxa efetiva (a.a.)"])
        commitment = float(row["comprometimento de renda"])
        min_value_field = row.get("valor mínimo", None)

        ins = get_insurance_rates(bank, age)

        sim = simulate_potential_row(
            amortization=amortization,
            quota=quota,
            term_months=term,
            annual_rate=annual_rate,
            commitment_rate=commitment,
            income=req.monthly_income,
            dfi_rate=ins["dfi_rate"],
            mip_rate=ins["mip_rate"],
            min_value_field=min_value_field,
        )

        if bank not in results_by_bank:
            results_by_bank[bank] = {"bank": bank, "PRICE": None, "SAC": None}
        results_by_bank[bank][amortization] = sim

    results = list(results_by_bank.values())

    return {"ok": True, "age": age, "operation": req.operation, "results": results}
