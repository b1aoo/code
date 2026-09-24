# EvoPARA

**Error-driven prompt optimization and problem-aligned retrieval for automated essay feedback.**

EvoPARA generates specific, rubric-aligned feedback on student essays across three dimensions: **content and topic adherence**, **structural coherence**, and **language expression**. It combines reusable expert assessment guidance with examples that address the problems in the current essay, without updating language-model parameters.

## Method overview

Useful corrective feedback requires both knowing **how to assess an essay** and finding **examples relevant to its weaknesses**. Broad rubric instructions can leave the assessment procedure implicit. Topic-based retrieval alone may return high-scoring essays whose strengths offer little guidance for correcting the target essay.

EvoPARA addresses these two needs through:

- **Error-Driven Prompt Optimization (EDPO):** uses recurring errors on validation essays to refine explicit assessment instructions. The resulting guidance helps the model inspect writing systematically and explain problems with concrete textual evidence.
- **Problem-Aligned Retrieval Agent (PARA):** provides support for individual essays through complementary content and problem retrieval channels. It uses essay diagnoses to seek relevant expert-annotated examples rather than relying on topic similarity alone.

The scripts in this repository cover feedback generation, retrieval variants, data preparation, and workflow-based evaluation. `RAG_EDPO.py` uses assessment instructions embedded in the script together with retrieved examples; it is a generation entry point, not a standalone implementation of the entire iterative prompt-optimization loop.

## Repository guide

| File | Purpose |
| --- | --- |
| [`RAG_EDPO.py`](RAG_EDPO.py) | Feedback generation with EDPO-style assessment guidance and content/problem retrieval. |
| [`RAG_dual_stage.py`](RAG_dual_stage.py) | Combined content and problem retrieval variant. |
| [`RAG_content_only.py`](RAG_content_only.py) | Content-based retrieval baseline. |
| [`RAG_problem_only.py`](RAG_problem_only.py) | Problem-based retrieval variant. |
| [`zero_shot.py`](zero_shot.py) | Feedback generation without retrieved examples. |
| [`few_shot.py`](few_shot.py) | Feedback generation with supplied examples. |
| [`indexing.py`](indexing.py) | Builds Elasticsearch text and embedding indexes from JSONL records. |
| [`search_func.py`](search_func.py) | Retrieval queries, result handling, and ranking utilities. |
| [`llm_func.py`](llm_func.py) | Diagnosis, feedback, and workflow helper functions. |
| [`data_func.py`](data_func.py) | Essay loading, query generation, and annotation utilities. |
| [`input_dealing.py`](input_dealing.py), [`json_generate.py`](json_generate.py) | Data preparation scripts. |
| [`pinggu.py`](pinggu.py) | Batch evaluation through a configured Coze workflow. |
| [`aliyun.py`](aliyun.py), [`jiekou.py`](jiekou.py), [`open_router.py`](open_router.py) | Provider-specific generation scripts. |
| [`LLM_list.txt`](LLM_list.txt) | Model identifiers grouped by API provider. |
| [`before/`](before/) | Earlier experimental implementations. |

## Environment and configuration

The code uses Python, Elasticsearch, Sentence Transformers, OpenAI-compatible or provider-specific model APIs, and Coze workflows. The following packages are inferred from the source imports; the repository does not currently include a pinned environment:

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install requests python-dotenv numpy pandas sentence-transformers elasticsearch cozepy openai
```

Use an Elasticsearch server and Python client compatible with the query and indexing APIs in the scripts. Retrieval uses `sentence-transformers/all-MiniLM-L6-v2`; its embedding dimension must match the index mapping.

Before running an experiment, configure:

| Setting | Where to check |
| --- | --- |
| Essay input and intermediate paths | `data_func.py` and the selected entry-point script. |
| Generation output path | `output_dir` in the entry point. |
| Essay range and concurrency | `START_NUM`, `END_NUM`, and `PARALLEL_NUM`. |
| API endpoint and model identifier | The selected entry point and any imported provider/workflow helpers. |
| Elasticsearch endpoint and index names | `indexing.py` and the retrieval entry points; the current endpoint is `http://localhost:9200`. |
| Index mappings and JSONL corpus | Paths in `indexing.py`. |
| Rubric or example files | `rule_path` and sample-file paths used by the selected script. |
| Evaluation workflow and output paths | `CONFIG` in `pinggu.py`. |

The scripts contain Windows paths rooted at `D:\zuowen`. Update all relevant paths, including those in shared helper modules. Changing only the `root_dir` argument does not change the input location used by `data_func.input_deal()`.

Copy [`.env.example`](.env.example) to `.env` in the repository root and fill in your own credentials locally. Scripts load this file through `python-dotenv`; existing environment variables take precedence. `.env` and Python bytecode caches are excluded from Git.

| Environment variable | Service |
| --- | --- |
| `DASHSCOPE_API_KEY` | DashScope generation and related helpers. |
| `COZE_API_TOKEN` | Coze clients and evaluation workflows, including shared helper imports. |
| `JIEKOU_API_KEY` | The provider used by `jiekou.py`. |
| `OPENROUTER_API_KEY` | OpenRouter generation. |

Keep credentials out of version control. Configure the endpoints, workflow IDs, and model names separately in the scripts. Model identifiers and workflow availability must be checked against the services you actually use.

## Data format

The associated PRISM benchmark contains 653 Chinese middle-school essays with expert feedback. The experiment split comprises 553 retrieval examples, 50 validation essays for prompt optimization, and 50 test essays. In the original numbering, IDs 1–50 are test essays, 51–100 are validation essays, and 101–653 form the retrieval corpus. Keep these partitions separate when preparing indexes and optimizing prompts.

**Essay data, teacher annotations, rubric files, Elasticsearch mapping files, and Coze workflow definitions are not bundled in this repository.** Prepare authorized inputs and the required service configuration before running the scripts.

### Per-essay input files

`data_func.input_deal()` reads UTF-8 text files with the prefix `202506` followed by a four-digit essay ID:

```text
2025060001-q.txt     # Essay topic / writing prompt
2025060001-s.txt     # Student essay
2025060001-ta.txt    # Teacher reference feedback
```

These reference annotations are loaded for the experimental evaluation pipeline. Keep the target essay's reference feedback out of its retrieval index and generation context.

### Retrieval corpus

`input_dealing.py` prepares JSONL records with the following fields. This example shows the schema only:

```json
{"id": 101, "title": "Essay topic", "content": "Essay text", "comment": "Expert feedback"}
```

`indexing.py` reads `title`, `content`, and `comment`, computes `text_embedding`, and writes documents to `research_index_bm25` and `research_index_knn`. It currently assigns sequential document IDs and processes the first JSONL file returned from its input directory; provide a dedicated corpus directory and check the resulting ID mapping.

## Running experiments

Run the scripts from the repository root after completing the configuration above. Several scripts initialize service clients during import, so even a baseline entry point can require Elasticsearch and model resources.

### 1. Build retrieval indexes

Supply the mapping files and retrieval-corpus JSONL path, then run:

```bash
python indexing.py
```

**This script deletes and recreates the configured indexes.** Use dedicated experiment indexes. The dense-vector mapping must match the embedding model, and the indexed corpus must exclude validation and test essays.

### 2. Generate feedback

Choose the experiment entry point, set a separate output directory, and configure the essay range:

```bash
# Generation with embedded assessment guidance and both retrieval channels
python RAG_EDPO.py

# Retrieval variants / baselines; run the variants needed for your experiment
python RAG_dual_stage.py
python RAG_content_only.py
python RAG_problem_only.py
python zero_shot.py
python few_shot.py
```

The scripts save a group of files for each essay: `-at.txt` for generated feedback, `-q.txt` for the topic, `-s.txt` for the essay, and `-ta.txt` for reference feedback. Batch failures are recorded in `error_log.txt`.

Input and generated-output filename formatting differ for single-digit IDs in the current scripts. The evaluator follows the generated-output pattern; preserve that pattern or update generation and evaluation together.

### 3. Evaluate feedback

Set `COZE_API_TOKEN` in your local `.env`. In `pinggu.py`, configure the workflow ID, input directory, output file, essay range, and concurrency. The workflow receives:

| Parameter | Content |
| --- | --- |
| `input0` | Essay topic followed by essay text. |
| `input1` | Teacher reference feedback. |
| `input2` | Generated feedback. |

Then run:

```bash
python pinggu.py
```

The script saves per-essay workflow responses. It requires an accessible workflow with the expected inputs and evaluation logic; workflow IDs in the source do not include the workflow definitions. Its configured results and error-log files are cleared at startup, so use separate paths for separate runs.

## Reproducing an experiment

Record the code revision, provider and exact model identifier, prompt, retrieval settings, data split, and evaluator configuration with each run. Use the same test essays and evaluation procedure when comparing generation methods. This repository provides the scripts; complete reproduction also requires the corresponding data, index mappings, prompts or external text assets, and workflow definitions described above.
