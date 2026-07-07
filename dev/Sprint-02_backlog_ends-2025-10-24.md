# Sprint 02 — Backlog (ends 2025-10-24)

## Feedback do Sprint 01
- O agrupamento estava a usar o **título**, mas deve passar a usar o **corpo da notícia** como base textual.  
- A seleção da **notícia líder** deve ser feita pelo **maior valor publicitário (AVP)**, e não pela data de publicação.  
- É necessário **avaliar se a métrica de cosseno** (TF-IDF + similaridade) realmente funciona bem ou se é preciso adotar outra abordagem.  
- O objetivo geral deve começar a focar-se na **transição da camada Bronze para Silver**, com dados limpos, deduplicados e prontos para análise.  
- Melhorar a **organização do trabalho e versionamento**, deixando de usar o Google Colab e passando a trabalhar em **GitHub** (branches, commits e PRs).

---

## Objetivo da sprint
- **Transitar Bronze → Silver** com dados limpos, deduplicados e com agrupamentos anotados.  
- **Evoluir o agrupamento** para usar o **corpo** (não o título) e escolher **líder por valor publicitário**.  
- **Avaliar** se TF-IDF + **similaridade de cosseno** é suficiente; comparar com alternativas - BERT e clustering (Kmeans, HDBSCAN).  

---

## Tarefas

- **Atualização da seleção de líder**: em cada grupo, marcar como líder a notícia com maior valor_publicitario

- **Testes de clustering**:

     KMeans sobre embeddings SBERT: testar MiniBatchKMeans e medir silhouette; analisar os resultados globais.

     HDBSCAN: experimentar em TF‑IDF/embeddings para descobrir k automaticamente 

- **Organização no repositório**: criar notebook/script em dev/

## Definição de Feito

- Código funcional de TF‑IDF + similaridade por threshold a gerar grupos e líder por grupo.

- Registo dos testes com SBERT+KMeans e HDBSCAN nos issues, incluindo notas de avaliação e decisão de manter TF‑IDF+threshold para a entrega.

- Repositório atualizado com notebook/script da sprint
