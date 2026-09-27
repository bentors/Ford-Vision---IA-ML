"""API de scoring de perfil de retenção — Ford Vision (IA & ML).

Executar a partir da raiz do repositório:
    uvicorn api.main:app --reload

Variáveis de ambiente (opcionais):
    FORD_VISION_MODEL_PATH  caminho do pipeline .joblib (padrão: models/perfil_retencao_pipeline.joblib)
    FORD_VISION_API_KEY     se definida, exige o header X-API-Key em /predict e /predict/batch
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import sklearn
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from src.features import PERFIL_INFO, PESO_RISCO, montar_features

logger = logging.getLogger("ford_vision.api")

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("FORD_VISION_MODEL_PATH", ROOT / "models" / "perfil_retencao_pipeline.joblib"))
METADATA_PATH = MODEL_PATH.with_name("metadata.json")

estado: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Modelo não encontrado em {MODEL_PATH}. Execute o notebook (seção 11) para gerá-lo."
        )
    estado["modelo"] = joblib.load(MODEL_PATH)
    estado["metadata"] = (
        json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}
    )
    versao_treino = estado["metadata"].get("sklearn_version")
    if versao_treino and versao_treino != sklearn.__version__:
        logger.warning(
            "scikit-learn diferente do treino (treino=%s, serviço=%s): retreine o modelo "
            "executando o notebook no ambiente do serviço.",
            versao_treino, sklearn.__version__,
        )
    yield
    estado.clear()


app = FastAPI(
    title="Ford Vision — Scoring de Retenção no Pós-Venda",
    description=(
        "Classifica o perfil de pós-venda de um veículo (Fiel, Esquecido ou Abandono) a partir de "
        "dados cadastrais e devolve um score de risco e uma ação sugerida. "
        "Modelo de **scoring da base ativa**: não é uma previsão no Dia Zero da venda."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def verificar_chave(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Se FORD_VISION_API_KEY estiver definida, exige o header X-API-Key correspondente."""
    esperada = os.getenv("FORD_VISION_API_KEY")
    if esperada and x_api_key != esperada:
        raise HTTPException(status_code=401, detail="API key ausente ou inválida.")


# ------------------------------------------------------------------ schemas
class Veiculo(BaseModel):
    modelo: str = Field(..., min_length=1, examples=["RANGER"], description="Nome do modelo (ex.: RANGER)")
    ano_modelo: int = Field(..., ge=2000, le=2040, examples=[2022])
    data_venda: date
    data_entrega: date
    data_inicio_garantia: date
    data_referencia: Optional[date] = Field(
        default=None, description="Data de pontuação; padrão: hoje."
    )


class Predicao(BaseModel):
    perfil: str
    probabilidades: dict[str, float]
    score_risco: float = Field(..., description="Risco esperado em [0, 1] (pesos por perfil = premissa de negócio)")
    nivel_risco: str
    acao_sugerida: str


class Lote(BaseModel):
    veiculos: list[Veiculo] = Field(..., min_length=1, max_length=1000)


class PredicaoLote(BaseModel):
    total: int
    predicoes: list[Predicao]


# ------------------------------------------------------------------ lógica
def _prever(veiculos: list[Veiculo]) -> list[Predicao]:
    modelo = estado["modelo"]
    df = pd.DataFrame([v.model_dump() for v in veiculos])
    X = montar_features(df, date.today())

    proba = modelo.predict_proba(X)
    classes = list(modelo.classes_)
    saidas = []
    for linha in proba:
        probs = {c: round(float(p), 4) for c, p in zip(classes, linha)}
        perfil = classes[int(linha.argmax())]
        score = sum(probs[c] * PESO_RISCO[c] for c in classes)
        saidas.append(
            Predicao(
                perfil=perfil,
                probabilidades=probs,
                score_risco=round(score, 4),
                nivel_risco=PERFIL_INFO[perfil]["nivel_risco"],
                acao_sugerida=PERFIL_INFO[perfil]["acao_sugerida"],
            )
        )
    return saidas


# ------------------------------------------------------------------ rotas
@app.get("/health", tags=["infra"])
def health():
    return {"status": "ok", "modelo_carregado": "modelo" in estado}


@app.get("/model-info", tags=["infra"])
def model_info():
    return estado.get("metadata", {})


@app.post("/predict", response_model=Predicao, tags=["scoring"], dependencies=[Depends(verificar_chave)])
def predict(veiculo: Veiculo):
    return _prever([veiculo])[0]


@app.post("/predict/batch", response_model=PredicaoLote, tags=["scoring"], dependencies=[Depends(verificar_chave)])
def predict_batch(lote: Lote):
    predicoes = _prever(lote.veiculos)
    return PredicaoLote(total=len(predicoes), predicoes=predicoes)
