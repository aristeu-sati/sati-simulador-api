from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from math import pow
from datetime import date
from typing import Optional, Dict, Any

app = FastAPI()

def annual_to_monthly(annual_rate: float) -> float:
    return pow(1.0 + annual_rate, 1.0 / 12.0) - 1.0

def calc_age(birth_date_iso: str) -> int:
    y, m, d = map(int, birth_date_iso.split("-"))
    today = date.today()
    age = today.year - y
    if (today.month, today.day) < (m, d):
        age -= 1
    return age

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

@app.get("/health")
def health():
    return {"ok": True}

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
    if req.property_value <= 0:
        raise HTTPException(status_code=422, detail="property_value must be > 0")
    if not (0 < req.max_quota <= 1):
        raise HTTPException(status_code=422, detail="max_quota must be between 0 and 1")
    if req.term_months <= 0:
        raise HTTPException(status_code=422, detail="term_months must be > 0")
    if req.monthly_income <= 0:
        raise HTTPException(status_code=422, detail="monthly_income must be > 0")
    if not (0 < req.commitment_rate <= 1):
        raise HTTPException(status_code=422, detail="commitment_rate must be between 0 and 1")

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
            "suggested_quota": 0.0,
            "min_down_payment_required": round(req.property_value, 2),
            "month1_installment_at_pv_allowed": round(B, 2),
        }

    pv_by_income = (max_installment - B) / A
    pv_by_ltv = req.property_value * req.max_quota

    pv_allowed = max(0.0, min(pv_by_income, pv_by_ltv))
    suggested_quota = pv_allowed / req.property_value
    min_down_payment = req.property_value - pv_allowed

    amort = pv_allowed / req.term_months
    interest = pv_allowed * i_m
    mip = pv_allowed * req.mip_rate
    installment = amort + interest + mip + B

    installment_at_max_quota = (
        (pv_by_ltv / req.term_months)
        + (pv_by_ltv * i_m)
        + (pv_by_ltv * req.mip_rate)
        + B
    )

    return {
        "ok": True,
        "fits": pv_allowed > 0,
        "max_installment_allowed": round(max_installment, 2),
        "pv_by_income": round(pv_by_income, 2),
        "pv_by_ltv": round(pv_by_ltv, 2),
        "pv_allowed": round(pv_allowed, 2),
        "suggested_quota": round(suggested_quota, 6),
        "min_down_payment_required": round(min_down_payment, 2),
        "month1_installment_at_pv_allowed": round(installment, 2),
        "fits_at_max_quota": installment_at_max_quota <= max_installment,
        "month1_installment_at_max_quota": round(installment_at_max_quota, 2),
        "components_at_pv_allowed": {
            "amortization": round(amort, 2),
            "interest": round(interest, 2),
            "mip": round(mip, 2),
            "dfi": round(B, 2),
        }
    }
