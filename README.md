# 🚗 Ford Vision — IA & Machine Learning para Retenção no Pós-Venda

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)

## 📌 Sobre o Projeto
Solução de IA/ML desenvolvida no **Desafio 02 — Impulsionando o VIN Share na América do Sul**, parceria acadêmica entre a **Ford Motor Company** e a **FIAP**. O objetivo é apoiar a retenção de clientes no pós-venda: identificar o perfil de cada veículo em relação à rede oficial de manutenção e gerar **leads proativos por concessionária**.

> **Enquadramento:** o modelo supervisionado é um **scoring da base ativa** — estima o perfil de pós-venda a partir de dados cadastrais, sem exigir o histórico de visitas. **Não é uma previsão no "Dia Zero" da venda** (a análise de sensibilidade quantifica a diferença).

## ⚙️ Metodologia
1. **Segmentação não supervisionada (K-Means, K=3)** sobre 175.137 veículos únicos (dados reais da rede Ford Brasil). Tratamentos: KM ≥ 1.000.000 removido, código de revisão sentinela (99/100) invalidado, winsorização (p99,9) e padronização. Perfis: *Cliente Fiel*, *Cliente Esquecido* e *Cliente de Abandono*. Silhouette 0,453.
2. **Classificação multiclasse** do perfil (rótulos derivados da segmentação) com 5 features cadastrais: modelo, ano-modelo, prazos de venda/entrega/garantia e tempo de garantia decorrido. Pré-processamento em `Pipeline` (sem vazamento estatístico).
3. **Comparação de 5 algoritmos** (Regressão Logística, Árvore, Random Forest, Gradient Boosting, MLP) por validação cruzada 5-fold, **ajuste de hiperparâmetros** (`RandomizedSearchCV`) e avaliação única no teste, sempre contra **baselines**.

## 📊 Resultados (conjunto de teste, 35.028 veículos)

| Modelo | Acurácia | F1 macro | ROC-AUC |
|---|---|---|---|
| Classe majoritária | 0,400 | 0,191 | 0,500 |
| Só `ano_modelo` | 0,791 | 0,736 | 0,886 |
| Só `tempo_garantia_ate_hoje` | 0,818 | 0,745 | 0,896 |
| **Gradient Boosting ajustado (final)** | **0,856** | **0,809** | **0,926** |

- Os 5 algoritmos ficaram em **empate técnico** (F1 macro CV entre 0,806 e 0,809): o teto é definido pelas features, não pelo algoritmo.
- **Por perfil:** Esquecido F1 0,97 · Abandono F1 0,86 · **Fiel F1 0,61** (a classe mais difícil sem dados comportamentais).
- **Sensibilidade:** sem idade/tempo de garantia ("Dia Zero"), o F1 macro cai de 0,809 para **0,585**.
- **Onde agrega valor:** nas coortes 2020–2023 (+3,7 a +28,1 p.p. sobre a classe majoritária da coorte). Em 2024–2026 quase todos os veículos são "Esquecidos" e o modelo apenas acompanha a classe majoritária.
- **Importância (permutação):** `tempo_garantia_ate_hoje` > `ModelName` > `ano_modelo`.

Limitações (viés de sobrevivência, pseudo-rótulos, censura temporal) e trabalhos futuros estão na **seção 10** do notebook.

## 📂 Estrutura do Repositório
```
notebooks/   IA&ML - Ford Vision - Sprint 3.ipynb  (notebook principal, já executado)
src/         features.py — engenharia de features compartilhada entre treino e API
api/         main.py — API FastAPI (/predict, /predict/batch, /health, /model-info)
tests/       test_api.py — testes automatizados da API
models/      perfil_retencao_pipeline.joblib + metadata.json
assets/      figures/ — gráficos gerados pelo notebook
docs/        Relatório executivo (PDF)
data/        (ignorado pelo Git) raw/ com o CSV da Ford e processed/ com as saídas
```

## ▶️ Como executar
```bash
pip install -r requirements.txt

# 1) Notebook: coloque vin_share_Desafio_02.csv em data/raw/ (dados da Ford, não versionados)
jupyter notebook "notebooks/IA&ML - Ford Vision - Sprint 3.ipynb"
#    Executar tudo regenera figuras, data/processed/ e models/

# 2) API (a partir da raiz do repositório)
uvicorn api.main:app --reload          # Swagger em http://127.0.0.1:8000/docs

# 3) Testes da API
pytest tests/ -v
```
Exemplo de chamada:
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "modelo": "RANGER", "ano_modelo": 2022,
  "data_venda": "2022-03-10", "data_entrega": "2022-03-25", "data_inicio_garantia": "2022-03-25"
}'
```
Se `FORD_VISION_API_KEY` estiver definida, a API exige o header `X-API-Key`.
> O `.joblib` depende da versão do scikit-learn (treinado com 1.8.0). Em outra versão, reexecute o notebook para regenerar o modelo.

## 👥 Equipe de Desenvolvimento
- Bento Rangel
- Ricardo Di Tilia
- Eric Yuji
- Kauê Pires
- Higor Batista
