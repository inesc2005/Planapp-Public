## 🏷️ O que é o *Medallion Architecture* (Sistema Medallion)

O *Medallion Architecture* é um padrão para organizar dados em camadas numa arquitetura de *lakehouse*, de forma a melhorar progressivamente a qualidade, estrutura e utilidade dos dados à medida que avançam das camadas iniciais para as finais.  

- [Documentação Microsoft (Azure Databricks)](https://learn.microsoft.com/en-us/azure/databricks/lakehouse/medallion)  
- [Glossário Databricks](https://www.databricks.com/glossary/medallion-architecture)  

As camadas típicas são:

| Camada | O que contém / faz | Para quem / uso principal |
|---|---------------------|----------------------------|
| **Bronze (Raw)** | Dados “crus”, obtidos diretamente da fonte. Sem limpeza ou processamento significativo. Preserva fidelidade, histórico, permite auditoria e reprocessamento. | Engenheiros de dados, operações, preparação de pipelines. |
| **Silver (Cleansed / Conformed)** | Dados limpos, validados, deduplicados, normalizados. Podem vir de várias fontes e serem unidos. “Just enough” transformação para tornar os dados mais utilizáveis. | Analistas, cientistas de dados, equipas que começam análises ou construções de modelos. |
| **Gold (Business / Serving Layer)** | Dados polidos para consumo final: agregações, modelos desnormalizados, tabelas para dashboards, relatórios e aplicações. Altamente otimizado para consulta. | Utilizadores de negócio, analistas finais, dashboards, apps de reporting. |

---

## 🔧 Como vamos aplicar isto no nosso projeto

Nós vamos usar esse padrão, mas com uma adaptação: vamos isolar também uma **Features Store** que ficará separada da camada Silver. Isto porque queremos ter um local dedicado de onde se obtêm atributos prontos para modelos de IA / machine learning, com garantias adicionais de qualidade, limpeza e preparação.

Aqui está como vamos encaixar cada peça:

- **Bronze**: já implementado — ingestão automática, dados crus em Postgres.
- **Silver**: vai ser onde limpamos (missing, valores errados), deduplicamos, extraímos entidades, categorizamos, etc.
- **Features Store**: vai preparar tabelas “ML-ready” com variáveis/colunas limpas, normalizadas, sem valores em falta, prontas para uso em modelos de IA. Embora os dados nela venham da Silver ou Bronze conforme for preciso, esta camada vai ser gerida separadamente para garantir consistência para treino/inferência.
- **Gold**: vai usar os dados da Silver e/ou Features Store para produzir tabelas de consumo, agregações, painéis, relatórios, etc.

---

## ✅ Benefícios esperados

- Maior qualidade dos dados à medida que passamos das camadas Bronze → Silver → Gold.  
- Auditabilidade e possibilidade de reprocessar com dados crus.  
- Separar a preparação de features (Features Store) facilita treino de modelos e garante que os dados de produção / inferência usam as mesmas transformações.  
- Possibilidade de comparar abordagens de IA com dados consistentes.  

---

## ⚠️ Limitações / cuidados

- Maior complexidade de pipeline (precisamos de automatizar bem as transições entre camadas).  
- Custos de armazenamento e computação crescentes à medida que mantemos Bronze + Silver + Features Store + Gold.  
- Definição clara de requisitos para Features Store — que variáveis precisamos, formato, periodicidade, etc.  

---

## 📌 Nota para o repositório

Vamos incluir este documento no repo para que fique claro o que se entende por cada camada, e em que fase está o nosso projeto. Também pode servir como referência para medir evolução através dos PDS (sprints).
