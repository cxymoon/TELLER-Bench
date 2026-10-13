# TELLER-Bench: A Benchmark for Evaluating LLM Agents on Banking Business Processes

## Overview

SOP-Bench evaluates the ability of LLM-based agents to execute complex, multi-step banking business workflows by making structured API tool calls in the correct order. Given a detailed Standard Operating Procedure (SOP) and a set of callable APIs, an agent must process each business case by invoking APIs in the exact sequence mandated by the SOP—without skipping, reordering, or prematurely terminating the call chain.

The benchmark is built around real banking business domains. This document uses **Mortgage Loan Origination (LOAN)** as the running example. The LOAN process covers three interdependent phases:

1. **Application Intake & Pre-qualification** — KYC verification, income/asset validation, credit report acquisition, pre-qualification screening, maximum loan amount calculation.
2. **Credit Underwriting** — Employment/income verification, credit analysis, DTI/DSCR calculation, probability-of-default scoring, credit grading, risk-adjusted interest rate determination.
3. **Property Valuation & Appraisal** — Appraisal order, comparable sales analysis, AVM cross-validation, LTV calculation, final decision rendering.

---

## Directory Structure

```
<project_root>/
├── prompts/                        # YAML prompt templates
│   ├── SOP_Generation.yaml
│   ├── Schema_Generation.yaml
│   ├── SOP_API_Generation.yaml
│   ├── SOP_API_Function_Code.yaml
│   ├── Dataset_Generation.yaml
│   ├── API_Graph_Generation.yaml
│   ├── Bench_API_Graph_Generation.yaml
│   ├── Function_Calling_Agentic_Prompt.yaml
│   └── ReAct_Agentic_Prompt.yaml
├── dataset_construct/
│   ├── dataset.py                  # Existing dataset (input records)
│   └── construct_dataset.py        # Script to expand the dataset via LLM
├── standard_benchmark/
│   ├── construct_benchmark.py      # Generate ground-truth API call sequences
│   └── benchmark_test.py           # Run benchmark against agent outputs
├── demo/
│   ├── sop_generation/             # SOP drafting demo
│   ├── schema_generation/          # Data schema generation demo
│   ├── api_generation/             # API spec generation demo
│   ├── api_function_code_generation/ # API function code generation demo
│   ├── dataset_generation/         # Synthetic dataset generation demo
│   ├── direct_graph_generation/
│   │   └── api_graph_generation.out  # Ground-truth API call sequence per record
│   └── agentic/
│       ├── function_calling_agentic.py  # Function-calling agent runner
│       └── MLN-*.txt               # Agent execution traces (one file per record)
└── graph/                          # Visualization scripts
```

---

## Environment Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install openai langchain langchain-openai langchain-core \
            pydantic pyyaml pandas transformers

# Set your API key and base URL (modify in scripts or export as environment variables)
export OPENAI_API_KEY="your_api_key_here"
export OPENAI_BASE_URL="https://your_api_endpoint/v1"
```

---

## Benchmark Construction Pipeline

The construction pipeline converts a business SOP into a complete benchmark: data schema → API definitions → synthetic dataset → ground-truth call sequences.

### Step 1: Prepare the SOP and Data Schema

Write the SOP as a structured document covering the business process (purpose, scope, definitions, input fields, main procedure, output fields). The SOP is the single source of truth for all downstream generation steps.

Use the prompt template `prompts/Schema_Generation.yaml` to let an LLM derive the data schema from the SOP. The schema distinguishes:

- **Input fields** — provided at application time (e.g., `applicant_name`, `gross_monthly_income`, `requested_loan_amount`).
- **Output fields** (wrapped in `<Field_to_be_obtained>`) — computed by API calls during processing (e.g., `debt_to_income_ratio`, `final_decision`). These remain empty in the raw dataset.

### Step 2: Generate API Specifications and Function Code

Use `prompts/SOP_API_Generation.yaml` to generate a structured list of APIs that map directly to SOP steps. Each API has a name, parameters, and a deterministic return value computable from the input record.

Use `prompts/SOP_API_Function_Code.yaml` to generate Python implementations of each API. The generated code is embedded in the agent execution scripts under `demo/agentic/`.

Key constraint: every API must be **deterministic**—given the same input record, it always returns the same output, so ground-truth answers can be computed offline.

### Step 3: Generate the Synthetic Dataset

`dataset_construct/construct_dataset.py` calls an LLM with the SOP, schema, and a seed dataset to generate 50 new synthetic records per batch. The prompt (`prompts/Dataset_Generation.yaml`) instructs the model to:

- Follow the exact schema format.
- Leave all `<Field_to_be_obtained>` fields empty.
- Start IDs from the current maximum to avoid collision.

Run dataset expansion:

```bash
cd dataset_construct/
python construct_dataset.py
```

The expanded dataset is stored in `dataset.py` as a list of records.

### Step 4: Generate Ground-Truth API Call Sequences

`standard_benchmark/construct_benchmark.py` sends the SOP, schema, dataset, and all API function implementations to an LLM, asking it to generate Python code that processes every record by calling APIs in the SOP-prescribed order.

The prompt instructs the model to:
- Execute **every** API in SOP order.
- Never skip steps, even if a pre-qualification decision is "Not Qualified."
- Output each field-value pair per record.

The resulting execution is captured in `demo/direct_graph_generation/api_graph_generation.out`. Each entry looks like:

```yaml
- application_id: "MLN-2025-048732"
  api_sequence:
    - api_name: "ValidateApplicantIdentity"
    - api_name: "ValidateEmploymentAndIncome"
    - api_name: "VerifyAssets"
    - api_name: "ObtainCreditConsent"
    - api_name: "PullTriBureauCreditReport"
    - api_name: "CalculateMiddleScore"
    - api_name: "ScreenPrequalification"
    - api_name: "CalculateMaxLoanAmount"
    - api_name: "VerifyIncomeUnderwriting"
    - api_name: "VerifyAssetsUnderwriting"
    - api_name: "AnalyzeCreditReport"
    - api_name: "CalculateDTI"
    - api_name: "ScoreProbabilityOfDefault"
    - api_name: "DetermineRiskAdjustedRate"
    - api_name: "EvaluateCompensatingFactors"
    - api_name: "OrderAppraisal"
    - api_name: "ConductSiteInspection"
    - api_name: "PerformComparableSalesAnalysis"
    - api_name: "ReconcileAppraisedValue"
    - api_name: "RunAVMCrossCheck"
    - api_name: "CalculateLTV"
    - api_name: "RenderFinalDecision"
```

Run ground-truth generation:

```bash
cd standard_benchmark/
python construct_benchmark.py
```

---

## Evaluation Pipeline

### Step 1: Run Agents on Benchmark Records

`demo/agentic/function_calling_agentic.py` runs an LLM agent over every record in the dataset. The agent is given:

- The SOP text (as system context).
- The data record for the current application.
- All API functions exposed as LangChain `StructuredTool` objects.
- A prompt template from `prompts/Function_Calling_Agentic_Prompt.yaml` instructing step-by-step SOP adherence.

The agent is built with LangChain's `create_tool_calling_agent` + `AgentExecutor`. For each record, `stdout` is captured via a `TeeOutput` wrapper and saved to a timestamped file `demo/agentic/MLN-<application_id>.txt`.

```bash
cd demo/agentic/
python function_calling_agentic.py
```

Each output file contains lines of the form:

```
> Invoking: `ValidateApplicantIdentity` with {"application_id": "MLN-2025-048732", ...}
> Invoking: `ValidateEmploymentAndIncome` with {...}
...
```

### Step 2: Stage 1 Evaluation — API Dependency Graph Similarity

Stage 1 evaluates the model's structural understanding of the SOP workflow by comparing the API dependency graph it reconstructs against the ground-truth graph.

`demo/eval/graph_similar/similar_cal.py` parses both graphs from their respective output files, then computes a weighted similarity score:

```
sim = α × node_overlap + β × edge_overlap + γ × path_overlap
```

where α = 0.4, β = 0.4, γ = 0.2 by default. It also reports node-level and edge-level precision, recall, and F1 separately.

Submit the job via SLURM (or run directly):

```bash
cd demo/eval/graph_similar/
sbatch similar_cal.sh
# or run locally:
python similar_cal.py
```

Output is written to `similar_cal.out`. Key reported metrics:

| Metric | Description |
|---|---|
| Node overlap | Jaccard similarity of node sets |
| Edge overlap | Jaccard similarity of edge sets |
| Path overlap | Jaccard similarity of all simple paths (cutoff = 4) |
| Node P / R / F1 | Precision, recall, F1 for API node selection |
| Edge P / R / F1 | Precision, recall, F1 for dependency edge prediction |

### Step 3: Stage 2 Evaluation — End-to-End Execution Accuracy

Stage 2 evaluates the model's end-to-end execution accuracy: whether the agent invokes the correct tools and produces the correct final outputs.

`demo/eval/final_score/score_cal.py` processes each agent trace file (`MLN-*.txt`) matched against `demo/eval/select/sampled_data.txt` (the 50-record evaluation subset), and for each record computes:

- **Tool call P / R / F1** — comparing the set of APIs the agent invoked against the standard tool list.
- **Pre-qualification decision accuracy** — whether the agent's `prequalification_decision` matches the ground-truth value from `standard_benchmark/benchmark_test.out`.
- **Final decision accuracy** — whether the agent's concatenated output across six key fields (`prequalification_decision`, `max_loan_amount_prequalified`, `estimated_interest_rate_prequalification`, `probability_of_default_score`, `compensating_factors_present`, `final_appraised_value`) matches the ground-truth `final_result`.

Per-record results are saved to `demo/agentic/analysis_results/<application_id>_result.txt`. A macro summary is written to `demo/agentic/analysis_results/analysis_summary.txt`.

Submit the job via SLURM (or run directly):

```bash
cd demo/eval/final_score/
sbatch score_cal.sh
# or run locally:
python score_cal.py
```

---

## Adding a New Business Domain

To add a new domain (e.g., Credit Card, Deposit, Investment):

1. Write the SOP document for the new domain.
2. Run schema generation (`prompts/Schema_Generation.yaml`).
3. Run API generation and code generation.
4. Generate and expand the dataset (`construct_dataset.py`).
5. Generate ground-truth sequences (`construct_benchmark.py`).
6. Create a new agent runner under `demo/agentic/`.
7. Copy `Process_indicators/loan_metrics.py` to `<domain>_metrics.py`, update `GT_FILE`, `MODELS`, `SKIP_APIS`, and the trace file glob pattern.

---

## Supported Models

The benchmark is model-agnostic. Any model accessible via an OpenAI-compatible API can be evaluated by pointing `base_url` and `model` in the agent runner to the target model. Current evaluation covers:

| Family | Models |
|---|---|
| Claude | claude4.5-sonnet, claude-haiku |
| GPT | gpt-5, gpt-4 |
| Gemini | Gemini-3, Gemini-2.5 |
| Qwen | Qwen3-32B, Qwen3-14B, Qwen3-8B, Qwen-max |
| DeepSeek | DeepSeek-R1, DeepSeek-V3 |
| Specialized agents | rStar2-Agent-14B, SkyRL-Agent-14B-v0 |
| Other | Kimi-K2, GLM-4.5 |
