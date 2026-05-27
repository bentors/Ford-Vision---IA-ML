# 🚗 Ford Vision - IA & Machine Learning para Retenção no Pós-Venda

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

## 📌 Sobre o Projeto
Este projeto é uma prova de conceito (PoC) desenvolvida durante o **Desafio: Impulsionando o VIN Share na América do Sul**, em parceria acadêmica entre a **Ford Motor Company** e a **FIAP (Faculdade de Informática e Administração Paulista)**. 

O objetivo é atuar na retenção de clientes no pós-venda utilizando algoritmos preditivos, permitindo que as concessionárias atuem preventivamente contra a evasão para o mercado mecânico independente logo no "Dia Zero" da venda do veículo.

## ⚙️ Metodologia e Tecnologias
A solução atua em duas frentes de Ciência de Dados:
1. **Modelagem Não Supervisionada (K-Means):** Segmentação de uma base histórica de 175.137 clientes únicos, otimizada via *Silhouette Score*, para mapeamento de perfis comportamentais.
2. **Classificação Preditiva (Random Forest):** Classificador operando sob rígida governança de dados (Zero *Data Leakage*), utilizando apenas 8 variáveis estáticas do momento do faturamento para classificar novos consumidores.

## 🚀 Principais Resultados Executivos
- **Diagnóstico Crítico:** O modelo mapeou que **78,7%** da base ativa possui risco latente de evasão.
- **Performance do Modelo:** Acurácia global de **83,16%**.
- **Detecção de Alta Sensibilidade:** Alcançamos um *Recall* impressionante de **98%** na identificação precoce do "Cliente Esquecido", o perfil com maior potencial de conversão e elasticidade a ações comerciais de baixo custo.
- **Explicabilidade:** O tempo de garantia restante provou ser o gatilho principal de evasão, concentrando **42% da relevância preditiva**.

## 📂 Estrutura do Repositório
- `/notebooks`: Contém o código fonte principal (`IA&ML - Ford Vision.ipynb`).
- `/docs`: Relatório executivo completo em PDF.
- `/assets/figures`: Visualizações, gráficos gerados e matrizes de confusão.

## 👥 Equipe de Desenvolvimento
- Bento Rangel
- Ricardo Di Tilia
- Eric Yuji
- Kauê Pires
- Higor Batista