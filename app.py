from fastapi import FastAPI
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
