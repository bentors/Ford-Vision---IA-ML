"""Engenharia de features compartilhada entre o notebook (treino) e a API (serving).

Manter as duas pontas usando a MESMA função evita *training-serving skew*:
o notebook valida a paridade entre a Base 2 construída inline e o resultado
de `montar_features` (ver seção "Deploy" do notebook).
"""
from __future__ import annotations

import pandas as pd

# Colunas de entrada esperadas por `montar_features`
COLUNAS_ENTRADA = [
    "modelo",
    "ano_modelo",
    "data_venda",
    "data_entrega",
    "data_inicio_garantia",
]

# Features consumidas pelo pipeline treinado (FS_BASE_ATIVA no notebook)
FEATURES_NUM = [
    "ano_modelo",
    "dias_venda_entrega",
    "dias_entrega_garantia",
    "tempo_garantia_ate_hoje",
]
FEATURES_CAT = ["ModelName"]
FEATURES = FEATURES_NUM + FEATURES_CAT

# Regras de negócio por perfil (espelham a tabela da seção 7 do notebook)
PERFIL_INFO = {
    "Cliente Fiel": {
        "nivel_risco": "Baixo",
        "acao_sugerida": (
            "Programa de fidelidade, agendamento antecipado automático "
            "e ofertas exclusivas."
        ),
    },
    "Cliente Esquecido": {
        "nivel_risco": "Médio",
        "acao_sugerida": (
            "Lembretes automáticos por KM/tempo, notificação push no app "
            "e agendamento com um clique."
        ),
    },
    "Cliente de Abandono": {
        "nivel_risco": "Alto",
        "acao_sugerida": (
            "Campanha de reativação via CRM, desconto progressivo na próxima "
            "revisão e contato proativo (WhatsApp/e-mail)."
        ),
    },
}

# Pesos usados no score de risco esperado = soma(P(perfil) * peso).
# São premissas de negócio a calibrar com a Ford (não foram aprendidas dos dados).
PESO_RISCO = {
    "Cliente Fiel": 0.0,
    "Cliente Esquecido": 0.5,
    "Cliente de Abandono": 1.0,
}


def montar_features(df: pd.DataFrame, data_ref=None) -> pd.DataFrame:
    """Converte dados cadastrais do veículo nas features do modelo.

    Parâmetros
    ----------
    df : DataFrame com as colunas de `COLUNAS_ENTRADA`. Se houver a coluna opcional
        `data_referencia`, ela tem prioridade linha a linha sobre `data_ref`.
    data_ref : data padrão de pontuação (str, date ou Timestamp). Padrão: hoje.
    """
    faltando = [c for c in COLUNAS_ENTRADA if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas ausentes: {faltando}")

    padrao = pd.Timestamp(data_ref) if data_ref is not None else pd.Timestamp.today().normalize()
    ref = pd.Series(padrao, index=df.index)
    if "data_referencia" in df.columns:
        ref = pd.to_datetime(df["data_referencia"], errors="coerce").fillna(ref)

    venda = pd.to_datetime(df["data_venda"], errors="coerce")
    entrega = pd.to_datetime(df["data_entrega"], errors="coerce")
    garantia = pd.to_datetime(df["data_inicio_garantia"], errors="coerce")

    out = pd.DataFrame(index=df.index)
    out["ano_modelo"] = pd.to_numeric(df["ano_modelo"], errors="coerce")
    out["dias_venda_entrega"] = (entrega - venda).dt.days.clip(lower=0)
    out["dias_entrega_garantia"] = (garantia - entrega).dt.days.clip(lower=0)
    out["tempo_garantia_ate_hoje"] = (ref - garantia).dt.days.clip(lower=0)
    out["ModelName"] = (
        df["modelo"].fillna("DESCONHECIDO").astype(str).str.upper().str.strip()
    )
    return out[FEATURES]
