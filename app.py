from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from math import pow
from datetime import date
from typing import Optional, Dict, Any, List, Tuple
import time
import os
import re

import pandas as pd
import requests


# =========================================================
# CONFIG (Google Sheets CSV + TTL cache)
# =========================================================

# Seus links públicos (CSV)
FIN_URL_DEFAULT = "https://docs.google.com/spreadsheets/d/1ONyZQvlJTIYSmHX4fGZTtH5j3N5yalMjrGMfPsyrEbE/gviz/tq?tqx=out:csv&gid=0"
INS_URL_DEFAULT = "https://docs.google.com/spreadsheets/d/1ONyZQvlJTIYSmHX4fGZTtH5j3N5yalMjrGMfPsyrEbE/gviz/tq?tqx=out:csv&gid=1836407575"

# TTL (segundos) - pode configurar no Render: CONFIG_TTL_SECONDS=300, por exemplo
CONFIG_TTL_SECONDS = int(os.getenv("CONFIG_TTL_SECONDS", "300"))

FIN_URL = os.getenv("FIN_URL", FIN_URL_DEFAULT)
INS_URL = os.getenv("INS_URL", INS_URL_DEFAULT)

# cache em memória
_last_load_ts: float = 0.0
_fin_df: Optional[pd.DataFrame] = None
_ins_df: Optional[pd.DataFrame] = None


def _now() -> float:
    return time.time()


def _to_bool(x: Any) -> bool:
    """
    Aceita TRUE/FALSE, true/false, 1/0, sim/não etc.
    """
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in ("true", "1", "sim", "yes", "y")


def _norm_col(s: str) -> str:
    return str(s).strip().lower()


def _fetch_csv_to_df(url: str) -> pd.DataFrame:
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Falha ao baixar CSV: {url} | erro: {e}")

    # pandas lê direto de string
    from io import StringIO
    return pd.read_csv(StringIO(r.text))


def load_configs(force: bool = False) -> None:
    """
    Carrega Financiamento e Seguros_export com cache TTL.
    """
    global _last_load_ts, _fin_df, _ins_df

    if (not force) and _fin_df is not None and _ins_df is not None:
        if (_now() - _last_load_ts) < CONFIG_TTL_SECONDS:
            return

    fin = _fetch_csv_to_df(FIN_URL)
    ins = _fetch_csv_to_df(INS_URL)

    # normaliza nomes de coluna
    fin.columns = [_norm_col(c) for c in fin.columns]
    ins.columns = [_norm_col(c) for c in ins.columns]

    # ---------------------------
    # Validação de colunas mínimas
    # ---------------------------
    fin_required = [
        "ativo",
        "banco",
        "operação",
        "amortização",
        "quota",
        "prazo máximo (meses)",
        "taxa efetiva (a.a.)",
        "comprometimento de renda",
    ]
    for c in fin_required:
        if c not in fin.columns:
            raise RuntimeError(f"Coluna obrigatória faltando em Financiamento: {c}")

    ins_required = ["ativo", "banco", "dfi_rate", "mip_rate", "idade"]
    for c in ins_required:
        if c not in ins.columns:
            raise RuntimeError(f"Coluna obrigatória faltando em Seguros_export: {c}")

    # ---------------------------
    # Filtra apenas ativos
    # ---------------------------
    fin["ativo_bool"] = fin["ativo"].apply(_to_bool)
    ins["ativo_bool"] = ins["ativo"].apply(_to_bool)

    fin = fin[fin["ativo_bool"] == True].copy()
    ins = ins[ins["ativo_bool"] == True].copy()

    # Tipos úteis
    fin["banco"] = fin["banco"].astype(str).str.strip()
    fin["operação"] = fin["operação"].astype(str).str.strip()
    fin["amortização"] = fin["amortização"].astype(str).str.strip().str.upper()

    ins["banco"] = ins["banco"].astype(str).str.strip()
    ins["idade"] = pd.to_numeric(ins["idade"], errors="coerce").fillna(0).astype(int)

    # ordena pra facilitar fallback de idade
    ins = ins.sort_values(["banco", "idade"]).reset_index(drop=True)

    _fin_df = fin
    _ins_df = ins
    _last_load_ts = _now()


# =========================================================
# Helpers Financeiros
# =========================================================

def annual_to_monthly(annual_rate: float) -> float:
    # taxa efetiva a.a. -> taxa efetiva a.m.
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
    """
    fator de pagamento PRICE (sem seguros): parcela = PV * k
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    if i_m <= 0:
        return 1.0 / n
    a = pow(1.0 + i_m, n)
    return (i_m * a) / (a - 1.0)


_money_re = re.compile(r"R\$\s*([\d\.\,]+)")


def parse_brl_money(s: Any) -> Optional[float]:
    """
    Aceita:
      - "R$ 200,00 (de Parcela)"
      - "R$750.000,00"
      - números
    """
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


def parse_float_any(v: Any) -> float:
    """Converte números vindos de CSV/planilha (pt-BR ou en-US) para float.
    Aceita exemplos:
      - "0,000055"  -> 0.000055
      - "1.234,56"  -> 1234.56
      - "1234.56"   -> 1234.56
      - 0.3 / 1 / 10
    """
    if v is None:
        return float("nan")
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s == "" or s.lower() == "nan":
        return float("nan")
    # Remove espaços
    s = s.replace(" ", "")
    # Se tiver vírgula, assume pt-BR (milhar com ponto e decimal com vírgula)
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    return float(s)


def is_min_installment_field(s: Any) -> bool:
    if s is None:
        return False
    return "parcela" in str(s).lower()


def brl(x: float) -> str:
    s = f"{x:,.2f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"


# =========================================================
# Seguro (DFI + MIP) por banco + idade
# =========================================================

def get_insurance_rates(bank: str, age: int) -> Optional[Dict[str, float]]:
    load_configs()  # respeita TTL
    if _ins_df is None:
        return None

    age_c = clamp_age(age)
    df = _ins_df[_ins_df["banco"].str.lower() == bank.strip().lower()]
    if df.empty:
        return None

    row = df[df["idade"] == age_c]
    if row.empty:
        # fallback: pega idade mais próxima <=, senão a menor disponível
        df2 = df.sort_values("idade")
        candidates = df2[df2["idade"] <= age_c]
        row = candidates.iloc[[-1]] if not candidates.empty else df2.iloc[[0]]

    r = row.iloc[0]
    return {"dfi_rate": parse_float_any(r["dfi_rate"]), "mip_rate": parse_float_any(r["mip_rate"])}


# =========================================================
# Simulação (potencial)
# =========================================================

def simulate_potential_row(
    amortization: str,  # "SAC", "PRICE", "MIX"
    quota: float,
    term_months: int,
    annual_rate: float,
    commitment_rate: float,
    income: float,
    dfi_rate: float,
    mip_rate: float,
    min_value_field: Any,
    bank: str | None = None,
) -> Dict[str, Any]:
    """
    Retorna o maior PV (financiamento) possível que caiba na renda,
    considerando seguros (DFI/MIP) e quota (LTV).

    Obs:
    - DFI calculado em cima do VALOR DO IMÓVEL => (PV/quota) * dfi_rate
    - MIP calculado em cima do PV => PV * mip_rate
    """
    if income <= 0:
        return {"ok": False, "fits": False, "error": "invalid_income", "message": "monthly_income must be > 0"}
    if not (0 < quota <= 1):
        return {"ok": False, "fits": False, "error": "invalid_quota", "message": "quota must be between 0 and 1"}
    if term_months <= 0:
        return {"ok": False, "fits": False, "error": "invalid_term", "message": "term_months must be > 0"}

    i_m = annual_to_monthly(annual_rate)
    max_installment = income * commitment_rate

    # DFI depende do valor do imóvel = PV/quota  => DFI = (PV/quota)*dfi_rate = PV*(dfi_rate/quota)
    dfi_per_pv = dfi_rate / quota

    amortization = (amortization or "").strip().upper()

    # ---- SAC
    if amortization == "SAC":
        # parcela = PV*(1/n + i_m + mip + dfi_per_pv)
        A = (1.0 / term_months) + i_m + mip_rate + dfi_per_pv
        pv_allowed = max_installment / A if A > 0 else 0.0

        amort = pv_allowed / term_months
        interest = pv_allowed * i_m
        mip = pv_allowed * mip_rate
        dfi = (pv_allowed / quota) * dfi_rate
        installment = amort + interest + mip + dfi

    # ---- PRICE
    elif amortization == "PRICE":
        k = price_factor(i_m, term_months)  # parcela sem seguros = PV*k
        A = k + mip_rate + dfi_per_pv
        pv_allowed = max_installment / A if A > 0 else 0.0

        payment_no_insurance = pv_allowed * k
        interest = pv_allowed * i_m
        amort = payment_no_insurance - interest
        mip = pv_allowed * mip_rate
        dfi = (pv_allowed / quota) * dfi_rate
        installment = payment_no_insurance + mip + dfi
    # ---- MIX (Itaú) — conforme documentação:
    #  - Meses 1..36: "PRICE base" (PMT do prazo total) com amortização acelerada (1.2x)
    #  - Mês 37+: vira SAC (não calculado aqui; aqui fazemos o potencial com base na 1ª parcela)
    elif amortization == "MIX":
        # Regra de elegibilidade (documentação: MIX apenas Itaú). Se bank não for informado, apenas calcula.
        if bank is not None:
            b = (bank or "").strip().lower()
            if ("itau" not in b) and ("itaú" not in b):
                return {
                    "ok": False,
                    "fits": False,
                    "error": "mix_only_itau",
                    "message": "Sistema MIX é exclusivo do banco Itaú (conforme documentação).",
                }

        k_price = price_factor(i_m, term_months)  # PMT base (sem seguros) = PV * k_price
        # 1ª parcela (sem seguros), com amortização acelerada:
        #   payment_base = PV*k_price
        #   interest1 = PV*i_m
        #   amort1 = (payment_base - interest1) * 1.2
        #   installment_no_insurance1 = interest1 + amort1 = PV*(1.2*k_price - 0.2*i_m)
        k_mix_month1 = (1.2 * k_price) - (0.2 * i_m)

        A = k_mix_month1 + mip_rate + dfi_per_pv
        pv_allowed = max_installment / A if A > 0 else 0.0

        payment_base = pv_allowed * k_price
        interest = pv_allowed * i_m
        amort = (payment_base - interest) * 1.2
        mip = pv_allowed * mip_rate
        dfi = (pv_allowed / quota) * dfi_rate
        installment = (interest + amort) + mip + dfi

    else:
        return {
            "ok": False,
            "fits": False,
            "error": "unsupported_amortization",
            "message": f"Unsupported amortization: {amortization}",
        }

    property_value = pv_allowed / quota
    down_payment = property_value - pv_allowed

    # Regra de mínimo (se existir)
    min_ok = True
    min_reason = None
    min_money = parse_brl_money(min_value_field)
    if min_money is not None:
        if is_min_installment_field(min_value_field):
            if installment < min_money:
                min_ok = False
                min_reason = f"installment<{min_money}"
        else:
            if property_value < min_money:
                min_ok = False
                min_reason = f"property_value<{min_money}"

    return {
        "ok": True,
        "fits": pv_allowed > 0 and min_ok,
        "min_rule_ok": min_ok,
        "min_rule_reason": min_reason,
        "amortization": amortization,
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
        },
    }


def build_summary_text(operation: str, age: int, income: float, results: List[Dict[str, Any]]) -> str:
    """
    Texto leigo e comparativo sempre.
    Mostra:
      - Melhor potencial (maior valor de imóvel) entre todos os bancos/sistemas que "fits"
      - Menor entrada (menor down_payment) entre todos que "fits"
      - Observações de bancos que ficaram fora (ex: sem seguro)
    """
    fits_items: List[Tuple[str, str, Dict[str, Any]]] = []
    not_ok_msgs: List[str] = []

    for item in results:
        bank = item.get("bank")
        for sys in ("PRICE", "SAC", "MIX"):
            r = item.get(sys)
            if r is None:
                continue
            if r.get("ok") is False:
                # erro do tipo missing_insurance / unsupported...
                msg = r.get("message") or r.get("error") or "erro"
                not_ok_msgs.append(f"• {bank} ({sys}): {msg}")
                continue
            if r.get("fits"):
                fits_items.append((bank, sys, r))

    lines = []
    lines.append(f"📌 *Simulação de Potencial*")
    lines.append(f"• Operação: {operation}")
    lines.append(f"• Idade: {age} anos")
    lines.append(f"• Renda mensal considerada: {brl(income)}")
    lines.append("")

    if not fits_items:
        lines.append("❌ No momento, *nenhuma opção* coube nas regras (renda/limites/seguros).")
        if not_ok_msgs:
            lines.append("")
            lines.append("⚠️ Observações:")
            lines.extend(not_ok_msgs[:8])
        return "\n".join(lines).strip()

    # melhor potencial = maior property_value
    best = max(fits_items, key=lambda t: float(t[2]["property_value"]))
    # menor entrada = menor down_payment
    low = min(fits_items, key=lambda t: float(t[2]["down_payment"]))

    b_bank, b_sys, b_r = best
    l_bank, l_sys, l_r = low

    lines.append("✅ *Melhor POTENCIAL (maior valor de imóvel)*")
    lines.append(f"• Banco: {b_bank} ({b_sys})")
    lines.append(f"• Imóvel até: {brl(b_r['property_value'])}")
    lines.append(f"• Entrada aprox.: {brl(b_r['down_payment'])}")
    lines.append(f"• Parcela inicial: {brl(b_r['month1_installment'])}")
    lines.append("")

    lines.append("✅ *Menor ENTRADA (menor valor de entrada)*")
    lines.append(f"• Banco: {l_bank} ({l_sys})")
    lines.append(f"• Imóvel até: {brl(l_r['property_value'])}")
    lines.append(f"• Entrada aprox.: {brl(l_r['down_payment'])}")
    lines.append(f"• Parcela inicial: {brl(l_r['month1_installment'])}")
    lines.append("")

    # Ranking curto (top 3 potenciais)
    fits_sorted = sorted(fits_items, key=lambda t: float(t[2]["property_value"]), reverse=True)[:3]
    lines.append("📊 *Top 3 por valor de imóvel*")
    for i, (bk, sys, rr) in enumerate(fits_sorted, start=1):
        lines.append(f"{i}) {bk} ({sys}) — Imóvel até {brl(rr['property_value'])} | Entrada {brl(rr['down_payment'])}")

    if not_ok_msgs:
        lines.append("")
        lines.append("⚠️ Bancos/opções ignorados por configuração:")
        lines.extend(not_ok_msgs[:8])

    return "\n".join(lines).strip()


# =========================================================
# API
# =========================================================

app = FastAPI(title="Sati Simulador API")


class PotentialRequest(BaseModel):
    monthly_income: float = Field(..., gt=0)
    birth_date: str = Field(..., description="YYYY-MM-DD")
    operation: str
    banks: Optional[List[str]] = None  # opcional: filtrar


class PropertySimulationRequest(BaseModel):
    bank: str = Field(..., description="Nome do banco (deve existir na configuração)")
    operation: str = Field(..., description="Operação (deve existir na configuração)")
    property_value: float = Field(..., gt=0, description="Valor do imóvel (R$)")
    monthly_income: float = Field(..., gt=0, description="Renda mensal (R$)")
    birth_date: str = Field(..., description="YYYY-MM-DD")
    amortization_system: Optional[str] = Field(
        None,
        description="Opcional. Se informado, força o sistema (SAC/PRICE/MIX). Se omitido, usa o configurado na operação/banco.",
    )



def month1_components(
    amortization: str,
    pv: float,
    term_months: int,
    annual_rate: float,
    quota: float,
    dfi_rate: float,
    mip_rate: float,
) -> Dict[str, float]:
    """Calcula componentes da 1ª parcela (mês 1) para um PV informado."""
    i_m = annual_to_monthly(annual_rate)
    amort = amortization.strip().upper()

    mip = pv * mip_rate
    dfi = (pv / quota) * dfi_rate  # DFI sobre valor do imóvel

    if amort == "SAC":
        amort_value = pv / term_months
        interest = pv * i_m
        installment = amort_value + interest + mip + dfi
        return {
            "amortization": amort_value,
            "interest": interest,
            "mip": mip,
            "dfi": dfi,
            "installment": installment,
        }

    if amort == "PRICE":
        k = (i_m * (1 + i_m) ** term_months) / ((1 + i_m) ** term_months - 1)
        installment_base = pv * k
        # no mês 1: juros = PV*i_m, amort = parcela_base - juros
        interest = pv * i_m
        amort_value = installment_base - interest
        installment = installment_base + mip + dfi
        return {
            "amortization": amort_value,
            "interest": interest,
            "mip": mip,
            "dfi": dfi,
            "installment": installment,
        }

    if amort == "MIX":
        # Regra explícita: mês 1 segue PRICE base + amortização acelerada (1.2x)
        k = (i_m * (1 + i_m) ** term_months) / ((1 + i_m) ** term_months - 1)
        installment_base = pv * k
        interest = pv * i_m
        amort_value = (installment_base - interest) * 1.2
        installment = amort_value + interest + mip + dfi
        return {
            "amortization": amort_value,
            "interest": interest,
            "mip": mip,
            "dfi": dfi,
            "installment": installment,
        }

    raise ValueError(f"Unsupported amortization system: {amortization}")


def pv_by_income_limit(
    amortization: str,
    quota: float,
    term_months: int,
    annual_rate: float,
    commitment_rate: float,
    income: float,
    dfi_rate: float,
    mip_rate: float,
    pv_search_cap: float,
) -> float:
    """Calcula o PV máximo pelo limite de renda (1ª parcela <= renda*comprometimento)."""
    i_m = annual_to_monthly(annual_rate)
    max_installment = income * commitment_rate
    amort = amortization.strip().upper()

    # Fórmulas fechadas quando possível
    if amort == "SAC":
        denom = (1 / term_months) + i_m + mip_rate + (dfi_rate / quota)
        if denom <= 0:
            return 0.0
        return max(0.0, max_installment / denom)

    if amort == "PRICE":
        k = (i_m * (1 + i_m) ** term_months) / ((1 + i_m) ** term_months - 1)
        denom = k + mip_rate + (dfi_rate / quota)
        if denom <= 0:
            return 0.0
        return max(0.0, max_installment / denom)

    # MIX: resolver por busca (monótono em PV)
    # Vamos buscar PV em [0, cap] onde cap é seguro (ex.: pv_by_ltv ou outro).
    lo, hi = 0.0, max(0.0, pv_search_cap)
    if hi == 0.0:
        return 0.0

    # garante que hi estoura (se não, aumenta até 4x cap ou até que estoure)
    def inst(pv: float) -> float:
        return month1_components(amort, pv, term_months, annual_rate, quota, dfi_rate, mip_rate)["installment"]

    if inst(hi) <= max_installment:
        # tenta aumentar hi até estourar, com limite
        for _ in range(6):
            hi2 = hi * 2
            if hi2 <= 0 or hi2 > pv_search_cap * 8:
                break
            if inst(hi2) > max_installment:
                hi = hi2
                break
            hi = hi2

    # binária
    for _ in range(60):
        mid = (lo + hi) / 2
        if inst(mid) <= max_installment:
            lo = mid
        else:
            hi = mid
    return lo

@app.get("/health")
def health():
    return {"ok": True, "ttl_seconds": CONFIG_TTL_SECONDS}


@app.get("/configs/operations")
def list_operations() -> Dict[str, Any]:
    load_configs()
    if _fin_df is None:
        return {"ok": False, "operations": []}
    ops = sorted(_fin_df["operação"].dropna().unique().tolist())
    return {"ok": True, "operations": ops}



@app.get("/configs/banks")
def list_banks(operation: Optional[str] = None) -> Dict[str, Any]:
    """Lista bancos disponíveis. Se 'operation' for informada, filtra bancos que possuem aquela operação."""
    load_configs()
    if _fin_df is None:
        return {"ok": False, "banks": []}

    df = _fin_df.copy()
    if operation:
        df = df[df["operação"].str.lower() == operation.strip().lower()]

    banks = sorted({str(b).strip() for b in df["banco"].dropna().tolist() if str(b).strip()})
    return {"ok": True, "banks": banks}

@app.post("/simulate/potential")
def simulate_potential(req: PotentialRequest) -> Dict[str, Any]:
    load_configs()  # usa cache TTL

    if _fin_df is None:
        raise HTTPException(status_code=500, detail="Config de financiamento não carregada")

    age = calc_age(req.birth_date)

    fin = _fin_df[_fin_df["operação"].str.lower() == req.operation.strip().lower()].copy()
    if fin.empty:
        raise HTTPException(status_code=422, detail=f"Nenhuma config para operation='{req.operation}' (e/ou todas estão inativas)")

    # Filtro opcional de bancos
    if req.banks:
        wanted = {b.strip().lower() for b in req.banks}
        fin = fin[fin["banco"].str.lower().isin(wanted)]
        if fin.empty:
            raise HTTPException(status_code=422, detail="Filtro de banks removeu todas as configs (ou ficaram inativas)")

    # Agrupa por banco e preenche PRICE/SAC/MIX
    results_by_bank: Dict[str, Dict[str, Any]] = {}

    for _, row in fin.iterrows():
        bank = str(row["banco"]).strip()
        amortization = str(row["amortização"]).strip().upper()

        # Garante chaves do banco
        if bank not in results_by_bank:
            results_by_bank[bank] = {"bank": bank, "PRICE": None, "SAC": None, "MIX": None}

        # Seguros
        ins = get_insurance_rates(bank, age)

        if ins is None:
            sim = {
                "ok": False,
                "fits": False,
                "error": "missing_insurance",
                "message": f"Banco sem seguros_export: {bank}",
            }
        else:
            quota = parse_float_any(row["quota"])
            term = int(parse_float_any(row["prazo máximo (meses)"]))
            annual_rate = parse_float_any(row["taxa efetiva (a.a.)"])
            commitment = parse_float_any(row["comprometimento de renda"])
            min_value_field = row.get("valor mínimo", None)

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
                bank=bank,
            )

        # mantém o último caso tenha duplicidade
        if amortization in ("PRICE", "SAC", "MIX"):
            results_by_bank[bank][amortization] = sim
        else:
            # amortização desconhecida cai como erro numa chave segura
            results_by_bank[bank][amortization] = {
                "ok": False,
                "fits": False,
                "error": "unsupported_amortization",
                "message": f"Unsupported amortization: {amortization}",
            }

    results = list(results_by_bank.values())

    # summary “objeto” (pra Make) — ignorando os que não cabem
    best_property = None
    lowest_entry = None

    for item in results:
        bank = item["bank"]
        for sys in ("PRICE", "SAC", "MIX"):
            r = item.get(sys)
            if not r or not isinstance(r, dict):
                continue
            if not r.get("ok") or not r.get("fits"):
                continue

            if best_property is None or r["property_value"] > best_property["property_value"]:
                best_property = {
                    "bank": bank,
                    "system": sys,
                    "property_value": r["property_value"],
                    "pv_financing": r["pv_financing"],
                    "down_payment": r["down_payment"],
                    "month1_installment": r["month1_installment"],
                }

            if lowest_entry is None or r["down_payment"] < lowest_entry["down_payment"]:
                lowest_entry = {
                    "bank": bank,
                    "system": sys,
                    "property_value": r["property_value"],
                    "pv_financing": r["pv_financing"],
                    "down_payment": r["down_payment"],
                    "month1_installment": r["month1_installment"],
                }

    summary_text = build_summary_text(req.operation, age, req.monthly_income, results)

    return {
        "ok": True,
        "age": age,
        "operation": req.operation,
        "summary": {
            "best_property_value": best_property,
            "lowest_down_payment": lowest_entry,
        },
        "summary_text": summary_text,   # <-- sempre preenchido (texto leigo)
        "results": results,
    }



@app.post("/simulate/property")
def simulate_with_property(req: PropertySimulationRequest) -> Dict[str, Any]:
    load_configs()  # usa cache TTL
    if _fin_df is None:
        raise HTTPException(status_code=500, detail="Config de financiamento não carregada")

    age = calc_age(req.birth_date)

    # Filtra config por banco + operação
    df = _fin_df.copy()
    df = df[df["banco"].str.lower() == req.bank.strip().lower()]
    df = df[df["operação"].str.lower() == req.operation.strip().lower()]
    if df.empty:
        raise HTTPException(
            status_code=422,
            detail=f"Nenhuma config para bank='{req.bank}' e operation='{req.operation}' (e/ou inativa).",
        )


    # Se o request informar amortization_system, restringe às configs compatíveis
    if req.amortization_system:
        df2 = df[df["amortização"].astype(str).str.strip().str.upper() == req.amortization_system.strip().upper()]
        if not df2.empty:
            df = df2

    # Se existirem múltiplas linhas para o mesmo banco/operação (ex.: modalidades/convênios),
    # escolhemos a linha mais "exigente" para renda (maior comprometimento) e, em empate,
    # a maior quota. Isso evita pegar uma linha mais restritiva por acidente (ex.: 15% vs 30%).
    def _safe_num(x):
        try:
            v = parse_float_any(x)
            return v if not (v != v) else float("-inf")  # NaN -> -inf
        except Exception:
            return float("-inf")

    if len(df) > 1:
        df = df.copy()
        df["_commitment_num"] = df["comprometimento de renda"].apply(_safe_num)
        df["_quota_num"] = df["quota"].apply(_safe_num)
        # ordena: maior comprometimento, maior quota, menor taxa efetiva (como desempate opcional)
        if "taxa efetiva (a.a.)" in df.columns:
            df["_rate_num"] = df["taxa efetiva (a.a.)"].apply(lambda x: _safe_num(x) if _safe_num(x)!=float("-inf") else float("inf"))
            df = df.sort_values(by=["_commitment_num", "_quota_num", "_rate_num"], ascending=[False, False, True])
        else:
            df = df.sort_values(by=["_commitment_num", "_quota_num"], ascending=[False, False])
        df = df.drop(columns=[c for c in ["_commitment_num","_quota_num","_rate_num"] if c in df.columns])

    row = df.iloc[0]


    bank = str(row["banco"]).strip()
    operation = str(row["operação"]).strip()

    # Sistema de amortização: força se vier no request, senão usa o configurado
    amort_cfg = str(row["amortização"]).strip().upper()
    amort = (req.amortization_system or amort_cfg).strip().upper()

    quota = parse_float_any(row["quota"])
    term = int(parse_float_any(row["prazo máximo (meses)"]))
    annual_rate = parse_float_any(row["taxa efetiva (a.a.)"])
    commitment = parse_float_any(row["comprometimento de renda"])
    min_value_field = row.get("valor mínimo", None)

    if not (0 < quota <= 1):
        raise HTTPException(status_code=500, detail=f"Quota inválida para {bank}/{operation}: {quota}")

    ins = get_insurance_rates(bank, age)
    if ins is None:
        raise HTTPException(status_code=422, detail=f"Banco sem seguros_export para cálculo: {bank}")

    dfi_rate = parse_float_any(ins["dfi_rate"])
    mip_rate = parse_float_any(ins["mip_rate"])

    # limites
    pv_by_ltv = req.property_value * quota
    pv_by_income = pv_by_income_limit(
        amortization=amort,
        quota=quota,
        term_months=term,
        annual_rate=annual_rate,
        commitment_rate=commitment,
        income=req.monthly_income,
        dfi_rate=dfi_rate,
        mip_rate=mip_rate,
        pv_search_cap=max(pv_by_ltv, 1.0),
    )

    pv_allowed = max(0.0, min(pv_by_ltv, pv_by_income))
    suggested_quota = pv_allowed / req.property_value if req.property_value > 0 else 0.0
    down_payment = req.property_value - pv_allowed

    comps_allowed = month1_components(amort, pv_allowed, term, annual_rate, quota, dfi_rate, mip_rate) if pv_allowed > 0 else {
        "amortization": 0.0, "interest": 0.0, "mip": 0.0, "dfi": 0.0, "installment": 0.0
    }

    comps_at_max_quota = month1_components(amort, pv_by_ltv, term, annual_rate, quota, dfi_rate, mip_rate)

    # Regra de mínimo (se existir)
    min_ok = True
    min_reason = None
    min_money = parse_brl_money(min_value_field)
    if min_money is not None:
        if is_min_installment_field(min_value_field):
            if comps_allowed["installment"] < min_money:
                min_ok = False
                min_reason = f"installment<{min_money}"
        else:
            if req.property_value < min_money:
                min_ok = False
                min_reason = f"property_value<{min_money}"

    fits = pv_allowed > 0 and min_ok and (comps_allowed["installment"] <= req.monthly_income * commitment + 1e-6)
    fits_at_max_quota = comps_at_max_quota["installment"] <= req.monthly_income * commitment + 1e-6

    i_m = annual_to_monthly(annual_rate)
    rate_effective_am = i_m
    rate_effective_aa = (1 + i_m) ** 12 - 1
    rate_nominal_aa = i_m * 12

    amort_plus_interest = comps_allowed["amortization"] + comps_allowed["interest"]
    admin_fee = 0.0  # não modelado no backend (por ora)

    
    # Cenário principal desta rota: SEMPRE no limite da quota (LTV máximo)
    pv_main = pv_by_ltv
    down_payment_main = req.property_value - pv_main
    comps_main = comps_at_max_quota

    # Indicadores para encaixar na renda atual (quando renda < necessário)
    pv_fit_income = max(0.0, min(pv_by_income, pv_by_ltv))
    down_payment_fit_income = req.property_value - pv_fit_income
    comps_fit_income = month1_components(amort, pv_fit_income, term, annual_rate, quota, dfi_rate, mip_rate) if pv_fit_income > 0 else {
        "amortization": 0.0, "interest": 0.0, "mip": 0.0, "dfi": 0.0, "installment": 0.0
    }

    # Renda necessária para suportar o cenário de quota máxima
    max_installment = req.monthly_income * commitment
    required_income_for_max_quota = comps_main["installment"] / commitment if commitment > 0 else float("inf")

    # Flags
    income_sufficient_for_max_quota = comps_main["installment"] <= max_installment + 1e-6

    # Helpers de formatação (pt-BR) para mensagem
    def _brl(x: float) -> str:
        s = f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"R$ {s}"

    def _pct(x: float, d: int = 2) -> str:
        return f"{x*100:.{d}f}%".replace(".", ",")

    # Mensagem pronta para WhatsApp (texto leigo)
    if income_sufficient_for_max_quota:
        whatsapp_message = (
            f"📊 Simulação de financiamento — {bank} ({operation})\n\n"
            f"🏠 Valor do imóvel: {_brl(req.property_value)}\n"
            f"💰 Financiamento no limite permitido ({int(quota*100)}%):\n"
            f"• Valor financiado: {_brl(pv_main)}\n"
            f"• Entrada necessária: {_brl(down_payment_main)}\n"
            f"• Primeira prestação: {_brl(comps_main['installment'])}\n\n"
            f"✅ Com a sua renda atual, esse cenário é compatível com o comprometimento máximo permitido ({_pct(commitment)}).\n\n"
            f"ℹ️ Importante: esta é uma simulação inicial. Os valores finais podem variar após a análise completa do banco "
            f"(documentação, perfil e avaliação do imóvel).\n\n"
            f"Se quiser, posso te ajudar a comparar cenários ou ajustar valores para encontrar a melhor opção."
        )
    else:
        extra_entry = max(0.0, down_payment_fit_income - down_payment_main)
        whatsapp_message = (
            f"📊 Simulação de financiamento — {bank} ({operation})\n\n"
            f"🏠 Valor do imóvel: {_brl(req.property_value)}\n"
            f"💰 Financiamento no limite permitido ({int(quota*100)}%):\n"
            f"• Valor financiado: {_brl(pv_main)}\n"
            f"• Entrada necessária: {_brl(down_payment_main)}\n"
            f"• Primeira prestação: {_brl(comps_main['installment'])}\n\n"
            f"⚠️ Com a sua renda atual, esse cenário excede o comprometimento máximo permitido ({_pct(commitment)}).\n\n"
            f"Opções para seguir:\n"
            f"1) Manter o financiamento de {_brl(pv_main)} → renda necessária aprox.: {_brl(required_income_for_max_quota)}\n"
            f"2) Encaixar na sua renda atual → financia aprox.: {_brl(pv_fit_income)}\n"
            f"   • Entrada necessária: {_brl(down_payment_fit_income)} ( +{_brl(extra_entry)} vs. entrada mínima )\n"
            f"   • Primeira prestação estimada: {_brl(comps_fit_income['installment'])}\n\n"
            f"ℹ️ Importante: esta é uma simulação inicial. Os valores finais podem variar após a análise completa do banco."
        )

    return {
        "ok": True,

        # Neste endpoint, `fits` significa: renda suficiente para o cenário de quota máxima
        "fits": income_sufficient_for_max_quota,
        "fits_at_max_quota": income_sufficient_for_max_quota,

        "min_rule_ok": min_ok,
        "min_rule_reason": min_reason,

        # básicos (para UI)
        "bank": bank,
        "operation": operation,
        "amortization_system": amort,

        # taxas
        "rates": {
            "effective_aa": rate_effective_aa,
            "effective_am": rate_effective_am,
            "nominal_aa": rate_nominal_aa,
            "term_months": term,
            "quota": quota,
            "commitment_rate": commitment,
        },

        # valores do CARD (sempre no limite da quota)
        "values": {
            "amortization_plus_interest": comps_main["amortization"] + comps_main["interest"],
            "mip": comps_main["mip"],
            "dfi": comps_main["dfi"],
            "admin_fee": admin_fee,
            "installment": comps_main["installment"],
        },

        # resumo do CARD (sempre no limite da quota)
        "summary": {
            "property_value": float(req.property_value),
            "pv_allowed": pv_main,
            "down_payment_required": down_payment_main,
            "first_installment": comps_main["installment"],
        },

        # indicadores de encaixe
        "income_sufficient_for_max_quota": income_sufficient_for_max_quota,
        "required_income_for_max_quota": required_income_for_max_quota,
        "pv_fit_income": pv_fit_income,
        "down_payment_required_to_fit_income": down_payment_fit_income,
        "first_installment_at_pv_fit_income": comps_fit_income["installment"],

        # detalhes úteis (auditoria)
        "pv_by_income": pv_by_income,
        "pv_by_ltv": pv_by_ltv,

        # compatibilidade com campos anteriores
        "pv_allowed": pv_main,
        "suggested_quota": pv_main / req.property_value if req.property_value > 0 else 0.0,
        "min_down_payment_required": down_payment_main,
        "month1_installment_at_pv_allowed": comps_main["installment"],
        "month1_installment_at_max_quota": comps_main["installment"],

        "components_at_pv_allowed": {
            "amortization": comps_main["amortization"],
            "interest": comps_main["interest"],
            "mip": comps_main["mip"],
            "dfi": comps_main["dfi"],
        },

        "components_at_max_quota": {
            "amortization": comps_main["amortization"],
            "interest": comps_main["interest"],
            "mip": comps_main["mip"],
            "dfi": comps_main["dfi"],
        },

        # cenário que cabe na renda (se precisar)
        "components_at_pv_fit_income": {
            "amortization": comps_fit_income["amortization"],
            "interest": comps_fit_income["interest"],
            "mip": comps_fit_income["mip"],
            "dfi": comps_fit_income["dfi"],
            "installment": comps_fit_income["installment"],
        },

        # mensagem pronta
        "whatsapp_message": whatsapp_message,

        "inputs": {
            "monthly_income": float(req.monthly_income),
            "birth_date": req.birth_date,
            "age": age,
        },
    }
