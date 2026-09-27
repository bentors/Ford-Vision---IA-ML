"""Testes da API de scoring. Executar da raiz do repositório: pytest tests/ -v

Pré-requisito: models/perfil_retencao_pipeline.joblib gerado pela seção 11 do notebook.
"""
import math

import pytest
from fastapi.testclient import TestClient

from api.main import app

VEICULO_OK = {
    "modelo": "RANGER",
    "ano_modelo": 2022,
    "data_venda": "2022-03-10",
    "data_entrega": "2022-03-25",
    "data_inicio_garantia": "2022-03-25",
    "data_referencia": "2026-05-04",
}
PERFIS = {"Cliente Fiel", "Cliente Esquecido", "Cliente de Abandono"}


@pytest.fixture(scope="module")
def client():
    # `with` dispara o lifespan (carrega o modelo)
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "modelo_carregado": True}


def test_model_info_expoe_metadados(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    assert "features" in r.json() and "metricas_teste" in r.json()


def test_predict_sucesso(client):
    r = client.post("/predict", json=VEICULO_OK)
    assert r.status_code == 200
    corpo = r.json()
    assert corpo["perfil"] in PERFIS
    assert set(corpo["probabilidades"]) == PERFIS
    assert math.isclose(sum(corpo["probabilidades"].values()), 1.0, abs_tol=1e-2)
    assert 0.0 <= corpo["score_risco"] <= 1.0
    assert corpo["nivel_risco"] in {"Baixo", "Médio", "Alto"}
    assert corpo["acao_sugerida"]


def test_predict_veiculo_novo_e_antigo_tem_perfis_coerentes(client):
    novo = {**VEICULO_OK, "ano_modelo": 2026, "data_venda": "2026-03-01", "data_entrega": "2026-03-10",
            "data_inicio_garantia": "2026-03-10"}
    antigo = {**VEICULO_OK, "ano_modelo": 2019, "data_venda": "2019-03-01", "data_entrega": "2019-03-10",
              "data_inicio_garantia": "2019-03-10"}
    assert client.post("/predict", json=novo).json()["perfil"] == "Cliente Esquecido"
    assert client.post("/predict", json=antigo).json()["perfil"] != "Cliente Esquecido"


def test_predict_modelo_desconhecido_nao_quebra(client):
    r = client.post("/predict", json={**VEICULO_OK, "modelo": "MODELO_INEXISTENTE"})
    assert r.status_code == 200


def test_predict_payload_invalido_retorna_422(client):
    r = client.post("/predict", json={"modelo": "RANGER"})
    assert r.status_code == 422


def test_predict_data_invalida_retorna_422(client):
    r = client.post("/predict", json={**VEICULO_OK, "data_venda": "nao-e-data"})
    assert r.status_code == 422


def test_batch_sucesso(client):
    r = client.post("/predict/batch", json={"veiculos": [VEICULO_OK, {**VEICULO_OK, "modelo": "TERRITORY"}]})
    assert r.status_code == 200
    corpo = r.json()
    assert corpo["total"] == 2 and len(corpo["predicoes"]) == 2


def test_batch_vazio_retorna_422(client):
    assert client.post("/predict/batch", json={"veiculos": []}).status_code == 422


def test_batch_acima_do_limite_retorna_422(client):
    r = client.post("/predict/batch", json={"veiculos": [VEICULO_OK] * 1001})
    assert r.status_code == 422


def test_api_key_exigida_quando_configurada(client, monkeypatch):
    monkeypatch.setenv("FORD_VISION_API_KEY", "segredo-de-teste")
    assert client.post("/predict", json=VEICULO_OK).status_code == 401                       # sem chave
    assert client.post("/predict", json=VEICULO_OK, headers={"X-API-Key": "errada"}).status_code == 401
    ok = client.post("/predict", json=VEICULO_OK, headers={"X-API-Key": "segredo-de-teste"})
    assert ok.status_code == 200


def test_health_nao_exige_chave(client, monkeypatch):
    monkeypatch.setenv("FORD_VISION_API_KEY", "segredo-de-teste")
    assert client.get("/health").status_code == 200
