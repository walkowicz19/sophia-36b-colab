# 🏛️ Kairos Sophia 2.0 MoE 36B — Treinamento & Avaliação no Google Colab Pro

Repositório oficial com o pipeline completo para treino e avaliação do modelo **Kairos Sophia 2.0 MoE 36B** (`k2_horizon`, 36B total / 4B ativos).

## Estrutura do Repositório
- `data/sophia-36b-seed-5k.jsonl.gz`: Dataset de validação e seed (5k exemplos).
- `data/sophia-36b-train-149k.jsonl.gz`: Corpus master de treinamento SFT (149k exemplos curados).
- `data/slice_dpo_preference_pairs_10k.jsonl.gz`: Pares de preferência DPO.
- `suite_sophia_36b_private.json`: Suíte privada Gauntlet de 400 casos held-out.
- `colab_phase0_smoke_test.py`: Smoke Test de viabilidade e Gate G0.
- `colab_phase1_train.py`: Loop de treinamento SFT com QLoRA NF4 e early stopping.
- `evaluate_sophia_36b_gauntlet.py`: Avaliador Gauntlet e identidade (4.096 tokens de geração por padrão).

## Executar no Colab

1. Abra `colab_sophia.ipynb` na revisão do PR com runtime GPU e BF16 nativo. O perfil usa uma única GPU, NF4 e SDPA. O Colab não garante A100, VRAM suficiente ou permanência da VM.
2. Mantenha o mesmo checkout, GPU/perfil de software e parâmetros ao retomar. O cache local fica na VM e o diretório de snapshots no Google Drive deve ser exclusivo da execução, com um único escritor.
3. Rode a auditoria e o smoke test antes do treino. O gate mede forward, backward e optimizer, mas só certifica as amostras testadas. Se o perfil de treino mudar, rode o smoke novamente com o contexto correspondente.
4. Configure `HF_TOKEN` em Colab Secrets somente se quiser publicar cada novo melhor adaptador em um repositório privado.

O treino usa loss exclusivamente nos tokens assistant do template nativo, descarta exemplos acima do limite e registra as contagens. O holdout fixo prioriza famílias de prompts menos frequentes no corpus e as separa por normalização de números, além de excluir sobreposição exata por ID/prompt. Essa heurística detecta variações numéricas, com cobertura parcial de paráfrases. A auditoria dos dados originais encontra 5.263 linhas SFT em comum com a validação, 4.000 prompts SFT repetidos e 149.000 instruções de sistema com a identidade legada. A correção é aplicada durante a preparação. O DPO é auditado separadamente e fica fora deste treino SFT.

O orçamento inicial é de 2.000 optimizer steps, com parada antecipada por validation loss. A cada avaliação, o pipeline salva localmente, compara os hashes do snapshot no Drive e retém até três snapshots íntegros. Cada snapshot contém o estado completo de retomada e uma cópia do melhor adapter. Em interrupções abruptas, o trabalho desde o último snapshot confirmado pode ser perdido. Ao concluir, o melhor adaptador local fica em `sophia-36b-checkpoints/sophia-36b-final-adapter`. Ele contém pesos PEFT e manifesto. Merge e exportação GGUF são etapas separadas.

O Gauntlet executa inferência real, registra o hash do adapter, diagnósticos por pilar, validação estrutural de diffs e perguntas de identidade em inglês e português. A suíte atual não inclui repositórios ou testes executáveis para comprovar correção funcional de todos os casos. O autoteste sintético grava um relatório separado e sem pontuação. A comparação com Sophia 9B Q5_K_M requer uma execução separada com a mesma suíte e artefatos de avaliação equivalentes.

A revisão K2 está fixada em `sophia_model.py`. As versões do bootstrap precisam passar no smoke CUDA. A retomada rejeita diferenças de corpus, código e versões registradas. Para mudar o orçamento ou outro parâmetro, use um novo diretório de execução.
