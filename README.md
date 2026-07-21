# Análise de Churn — Telco Customer Churn (Databricks + PySpark)

Pipeline de dados em arquitetura medallion (Bronze → Silver → Gold) para análise e previsão de churn de clientes de telecomunicações, com modelo preditivo e dashboard interativo.

📊 **Dashboard:** [Data Studio](https://datastudio.google.com/reporting/858532ac-1c6e-440c-a5d5-9a7653e368df)

## Objetivo

Simular um cenário real de engenharia e análise de dados: ingestão de dados brutos, tratamento em camadas, modelagem preditiva e visualização de negócio, usando um stack orientado a escala (PySpark/Databricks), diferente da abordagem tradicional de Excel/Power BI.

## Stack

- **PySpark** (processamento distribuído)
- **Databricks** (ambiente de execução, Delta Lake, Unity Catalog)
- **Spark SQL** (análises)
- **MLlib** (regressão logística)
- **Data Studio** (dashboard)

## Dataset

[Telco Customer Churn (IBM)](https://www.kaggle.com/datasets) — 7.043 clientes, 50 colunas, incluindo dados demográficos, de uso de serviço, cobrança e motivo de cancelamento.

## Arquitetura

```
Bronze  → ingestão bruta da tabela original (Unity Catalog), sem transformação
Silver  → limpeza, tratamento de nulos, padronização de tipos e nomes de coluna
Gold    → features de negócio (segmentação por tempo de contrato) e agregações
```

Essa separação em camadas segue o padrão *medallion architecture*, comum em ambientes de dados modernos (data lakehouse), e permite rastreabilidade e reprocessamento a qualquer momento sem perder o dado original.

## Modelo preditivo

Regressão logística (Spark MLlib) prevendo probabilidade de churn a partir de tempo de contrato, cobrança mensal, tipo de contrato, serviço de internet e método de pagamento.

**AUC: 0.84**

## Principais insights

- Clientes com **contrato mensal** e **até 12 meses de casa** apresentam taxa de churn de até **61%**, contra menos de **5%** em contratos de dois anos com mais de 48 meses.
- **Concorrência** é o principal motivo de cancelamento relatado, à frente de atitude de atendimento e insatisfação.
- Clientes que pagam por **cheque/boleto** têm taxa de churn superior aos que usam **cartão de crédito**.

## Nota metodológica

Na agregação por segmento, a taxa de churn foi calculada como `SOMA(churn) / SOMA(clientes)` (taxa ponderada), em vez da média simples das taxas por subgrupo, evitando distorção de segmentos com poucos clientes.

## Estrutura do repositório

```
notebooks/
  01_churn_medallion_pipeline.py       # notebook Databricks (Bronze/Silver/Gold + SQL + modelo)
data/
  gold_churn_summary_export.csv        # tabela agregada usada no dashboard
  gold_telco_churn_export.csv          # tabela completa (nível cliente)
README.md
```

## Autor

Petrus Sampaio — [LinkedIn](https://www.linkedin.com/in/petrus-sampaio-6b3b8924a/) · [GitHub](https://github.com/petrussampaio)
