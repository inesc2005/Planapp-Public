# Sprint 01 — Backlog (ends 2025-10-07)

## Objetivo da sprint
- Expor dados da **camada Bronze** de forma acessível (ex.: via API simples).
- Criar um protótipo inicial para **agrupar notícias semelhantes** e identificar um **líder por grupo**.
- Explorar alternativas possíveis para melhorar esse agrupamento.

---

## Tarefas

1. **Acesso à Bronze via API**  
   - Implementar um endpoint simples (ex.: FastAPI/Flask) que permita obter as notícias armazenadas na tabela `cision_news`.
   - O script de ingestão inicial que acede à fonte de dados (Supabase) está em `src/ingest.py`.
   - Suportar filtros básicos: por data de publicação e por órgão.  

2. **Agrupamento de notícias semelhantes**  
   - Usar um método simples (por exemplo, TF-IDF + similaridade de cosseno) para agrupar notícias muito parecidas.  
   - Em cada grupo, selecionar uma notícia como **líder** (ex.: a mais longa ou a primeira publicada).  

3. **Proposta de alternativas**  
   - Preparar um documento curto com pelo menos **duas abordagens possíveis** para este problema (ex.: embeddings, clustering, MinHash/LSH).  
   - Indicar vantagens/desvantagens de cada uma (custo, complexidade, qualidade esperada).  

---

## Definição de Feito
- Endpoint a devolver notícias da Bronze com filtros básicos.  
- Protótipo a gerar grupos de notícias com líder identificado em cada grupo.  
- Documento com alternativas técnicas entregue.  
