# 📰 Clipping de Notícias com Arquitetura Medallion

Este repositório contém o **projeto académico aplicado da Licenciatura em Tecnologias Digitais e Inteligência Artificial (TDIA)**, cujo objetivo é desenvolver uma pipeline de processamento de notícias baseada na arquitetura **Medallion (Bronze, Silver, Gold)**.  

A atividade enquadra-se no programa de experimentação com **metodologias de Inteligência Artificial (IA)** para melhorar a **avaliação e classificação de notícias**.

---

## 🎯 Objetivos

- Criar uma pipeline que permita evoluir da ingestão inicial (Bronze) até à camada Gold.  
- Explorar técnicas de **deduplicação de notícias**, **extração de entidades**, **classificação por taxonomia** e **criação de embeddings**.  
- Avaliar diferentes abordagens de IA, desde modelos locais a soluções pagas via API, comparando custos e desempenho.  

---

## 🏗️ Estado Atual

- Apenas a **camada Bronze** está implementada.  
- Ingestão de notícias feita automaticamente via **GitHub Actions**, gravando numa tabela Postgres (`cision_news`).  
- Estrutura da tabela inclui:
  - `id`, `titulo`, `noticia`, `link_text`, `data_publicacao`, `autores`, `categoria`, `orgao`, etc.

---

## 🔜 Próximos Passos

As próximas fases do projeto visam:

1. **Silver (Curated Tables)**  
   - Deduplicar notícias através de métodos de ML/NLP.  
   - Adicionar etiquetas (temas, políticas públicas, tópicos relevantes).  
   - Enriquecer com metadados adicionais (fontes, entidades, classificações automáticas).  
   - Garantir consistência e preparação para consumo na Features Store.  

2. **Features Store**  
   - Preparar tabelas “ML ready” com:  
     - Sem valores em falta.  
     - Normalização de variáveis numéricas.  
     - Parâmetros prontos para modelos supervisionados e não supervisionados.  

3. **Gold (Serving Layer)**  
   - Construir tabelas para análise final e visualização:  
     - Tendências agregadas.  
     - Notícias em destaque.  
     - Séries temporais (agregadas por política, órgão, categoria).  
     - Tabelas de políticas públicas.  

4. **Atividades de IA**  
   - Agrupar notícias repetidas (clustering, BERT, LLM local).  
   - Classificar por taxonomia ao menor custo possível (comparação entre modelos locais e APIs externas).  
   - Gerar embeddings para vetorização semântica.  

---

## 📅 Organização em PDS (Sprints)

Cada **PDS (Ponto de Situação / Sprint)** será dividido em dois tipos de atividades:

1. **Desenvolvimento baseado em requisitos definidos**  
   - Implementação incremental das camadas (Silver, Features Store, Gold).  
   - Criação de scripts e pipelines reprodutíveis.  

2. **Preparação de propostas para soluções**  
   - Exploração de alternativas técnicas (ex.: diferentes modelos de classificação).  
   - Análise de custos vs. desempenho.  
   - Sugestões para evolução futura do sistema.  

---

## ⚙️ Tecnologias previstas

- **Postgres** → armazenamento.  
- **GitHub Actions** → ingestão automatizada.  
- **Python / ML / NLP** → clustering, classificação e embeddings.  
- **LLMs (locais e externos)** → suporte a classificação.  

---

## 📌 Nota

Este repositório representa um **ponto de partida**. A única componente operacional é a ingestão de dados para Bronze.  
Todo o restante desenvolvimento (Silver, Features Store, Gold, IA) será realizado pelas alunas no âmbito da atividade.  
