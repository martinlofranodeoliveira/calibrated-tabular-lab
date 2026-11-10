# Calibrated Tabular Lab

Versão de desenvolvimento 0.1.0. Pacote Python e CLI offline para experimentos tabulares com quatro conjuntos separados: treino, calibração de probabilidades, escolha do limiar e teste final.

O scikit-learn ajusta uma floresta fixa, uma regressão logística de referência e uma referência pela proporção das classes. A floresta congelada recebe calibração em dados reservados. O código original audita divisões determinísticas, verifica métricas e custos de limiares independentemente e gera relatórios locais limitados. A calibração pode piorar as métricas; o relatório preserva os valores originais e calibrados.

## Instalação e execução

Plataforma qualificada: Linux amd64 e CPython 3.13, com as dependências exatas dos arquivos de hashes. A execução não precisa de compilador, GPU, serviço, download de modelos ou API.

```sh
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install --require-hashes -r requirements.lock
python -m pip install --no-build-isolation --no-deps .
calibrated-tabular-lab demo --assert --output report-demo
python -m unittest discover -s tests -v
```

O diretório pai da saída deve existir. Um destino nunca é sobrescrito. Comandos normais emitem um resultado JSON; `--help` mostra ajuda textual. Confira a saída completa, o código de retorno e os hashes dos artefatos.

Os exemplos `stratified`, `group` e `chronological` usam os mesmos 320 registros e oito atributos sintéticos de desvio de processo. `group` separa lotes; `chronological` ordena intervalos e aplica embargo. São garantias diferentes: uma divisão temporal pode continuar compartilhando grupos.

```sh
calibrated-tabular-lab demo --assert --preset chronological --output report-time
```

## API Python

```python
from calibrated_tabular_lab import (
    ExperimentConfig, create_split_plan, make_synthetic_process_dataset,
    run_experiment,
)
dataset = make_synthetic_process_dataset()
config = ExperimentConfig(seed=42, fp_cost=1, fn_cost=4)
plan = create_split_plan(dataset, seed=42)
result = run_experiment(dataset, config=config, plan=plan)
print(result.test_evaluations["calibrated_forest"].brier_loss)
```

O plano congelado guarda identificadores canônicos. Ao reutilizá-lo com alterações somente nos rótulos do teste final, treino, calibração, probabilidades e política permanecem iguais. A estratificação inicial usa legitimamente os rótulos. Uma configuração explícita deve coincidir com estratégia, semente e embargo do plano; exigências adicionais são verificadas.

## Entrada e relatórios

O JSON fechado contém `rows` e, opcionalmente, `feature_names`. Cada registro tem `row_id`, `features`, `label` e, opcionalmente, `group_id`/`timestamp`; metadados devem ser omitidos ou ter seu tipo declarado. O [README em inglês](README.md#analyze-a-local-input) mostra como gerar uma entrada sintética completa pelo pacote instalado.

```sh
calibrated-tabular-lab analyze --input synthetic.json --seed 42 --output report-input
```

`--input -` lê stdin limitado. Chaves duplicadas, campos desconhecidos, números não finitos, tipos incompatíveis e entradas por link simbólico/FIFO são rejeitados. Não há expressões, plugins nem leitura de modelos serializados.

O resultado inclui JSON estrito com rótulos/probabilidades para verificação, CSV com todas as faixas de confiabilidade, CSV com todos os limiares, HTML independente com texto escapado e manifesto SHA256. O relatório não carrega recursos externos nem executa scripts.

## Limites e interpretação

São aceitos 160–4096 registros e 1–16 atributos finitos, de magnitude até 1.000.000, ou `null` para ausência. IDs são únicos, ASCII imprimível e limitados a 64 bytes. Cada conjunto precisa de pelo menos oito registros de cada classe. Entrada e relatório completo de cinco arquivos têm limites separados de 4 MiB. `--max-output-bytes` pode reduzir o limite de saída.

A floresta possui 64 árvores com profundidade máxima seis; a regressão logística usa até 200 iterações. Não há busca de parâmetros nem repetição de sementes. Penalidades são unidades hipotéticas, não dinheiro ou recomendação de implantação. Brier e log loss medem aspectos além da calibração; faixas com poucos dados não certificam probabilidades. Consulte o [contrato exato](docs/contract.md).

`INVALID_INPUT`, `INVALID_SPLIT`, `FIT_FAILED` e `UNKNOWN_BUDGET` permanecem distintos. Colisões, falhas anteriores à publicação e publicação visível com durabilidade incerta possuem estados próprios. Timeout, OOM e problemas de dependência são execução incompleta, não rejeições matemáticas. Não há garantia de tempo real, durabilidade após queda de energia ou proteção contra mutação hostil do sistema de arquivos.

Os exemplos são sintéticos, sem dados reais de clientes, decisões médicas, de crédito, contratação ou segurança crítica, alegação de prontidão produtiva ou certificação de acurácia.

## Verificação isolada

```sh
sh scripts/ci.sh
```

Apenas a aquisição oficial com hashes usa rede. Testes e consumidor instalado novo usam contêineres sem rede ou bind do host, UID/GID 1000, capabilities removidas, no-new-privileges, limite de 256 processos, 1 GiB e duas CPUs. O consumidor não tem checkout, pacote de build nem compilador. Evidência de CI hospedada depende de execução real.

A verificação exporta wheel, distribuição de fontes e relatórios sintéticos para o diretório ignorado `dist/`.

Código original MIT © 2026 [Lofrano Tecnologia](https://lofranotecnologia.com.br). Dependências mantêm licenças próprias: [avisos exatos](THIRD_PARTY_NOTICES.md).
