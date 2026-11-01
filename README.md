# 🏛️ Kairos Sophia 2.0 MoE 36B — Treinamento & Avaliação no Google Colab Pro

Repositório oficial com o pipeline completo para treino e avaliação do modelo **Kairos Sophia 2.0 MoE 36B** (`k2_horizon`, 36B total / 4B ativos).

## Estrutura do Repositório
- `data/sophia-36b-seed-5k.jsonl.gz`: Dataset de validação e seed (5k exemplos).
- `data/sophia-36b-train-149k.jsonl.gz`: Corpus master de treinamento SFT (149k exemplos curados).
- `data/slice_dpo_preference_pairs_10k.jsonl.gz`: Pares de preferência DPO.
- `suite_sophia_36b_private.json`: Suíte privada Gauntlet de 400 casos held-out.
- `colab_phase0_smoke_test.py`: Smoke Test de viabilidade e Gate G0.
- `colab_phase1_train.py`: Loop de treinamento SFT com QLoRA NF4 e early stopping.
- `evaluate_sophia_36b_gauntlet.py`: Avaliador com diretriz estrita de geração livre (`max_new_tokens >= 4096`).
