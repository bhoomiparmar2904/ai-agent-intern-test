# Aster & Row Support Agent

A reliability-focused RAG support agent for the fictional Aster & Row ecommerce company.

Answers policy questions with citations, looks up order status via a tool (never guesses), maintains multi-turn context, and refuses to be steered by instructions hidden in retrieved content or tool output.

## Setup (from a clean clone)

```bash
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env           # then edit .env and add your own GROQ_API_KEY
```

Create your own Groq API key and place it in `.env`.

## Run

```bash
python src/cli.py              # interactive chat
python src/cli.py --debug      # also prints retrieval + tool trace per turn
```

## Required environment variables

| Variable       | Required | Notes                                           |
| -------------- | -------- | ----------------------------------------------- |
| `GROQ_API_KEY` | yes      | Your own Groq API key. Never commit this value. |
| `GROQ_MODEL`   | no       | Defaults to `openai/gpt-oss-120b`.              |

## Model, embedding, framework, and storage choices

* **LLM**: `openai/gpt-oss-120b` via Groq's OpenAI-compatible API, accessed with the `openai` Python SDK using Groq's `base_url`. Standard OpenAI-style function/tool calling is used for the order-lookup tool.

* **Retrieval / embeddings**: TF-IDF (`scikit-learn`) cosine similarity over markdown chunks, built in-process at startup. This is not a hosted embedding API. It requires no additional API key or network dependency, is deterministic, and is fast enough for the 14-file knowledge base. The tradeoff is weaker semantic matching for paraphrases that share few literal words with the source text.

* **Framework**: None. The project uses a thin, explicit Python agent loop in `src/agent.py` rather than an agent framework, keeping retrieval, tool use, and control flow visible and testable.

* **Storage**: No persistent storage. The TF-IDF index and orders map are rebuilt in memory from `knowledge-base/` and `data/orders.json` on every process start. No vector database is required.

## Architecture

```text
knowledge-base/*.md ──► retrieval.py (chunk by ## heading, TF-IDF index)
data/orders.json     ──► tools.py (OrderLookupTool: allowlisted fields only)
                              │
user message ──► agent.py: retrieve top-k passages → build system context
                  → Groq (openai/gpt-oss-120b, system prompt + tools)
                  → tool call? → OrderLookupTool.call()
                  → tool result fed back → final answer + <CONTROL> block
                  → parsed into {answer, sources, handoff}
                              │
                  logging_utils.py ──► logs/*.jsonl (structured, no secrets)
```

Two design choices carry most of the reliability work:

1. **Field allowlisting happens in code, not in the prompt.** `OrderLookupTool.call()` only constructs a dictionary containing safe fields. The raw order, including `internal.*` fields and customer PII, never reaches the model. Therefore, prompt injection inside `warehouse_note` cannot be used to expose those fields.

2. **Status-precedence logic is computed in code.** Whether `carrier` and `estimated_delivery` should be shown is decided by `tools.py` from the order `status` before the result reaches the model. This prevents the model from incorrectly displaying stale delivery information for cancelled or returned orders.

## Running evaluations

```bash
python evaluation/run_eval.py
python evaluation/run_eval.py --verbose
```

The evaluation suite uses deterministic assertions including substring checks, keyword-overlap concept checks, required tool calls and arguments, required/forbidden cited sources, and the `handoff` flag emitted by the agent. The evaluation is not graded by a second LLM.

## Final evaluation results

The final evaluation run achieved:

**22/22 tests passed**

| Category               |    Result |
| ---------------------- | --------: |
| Retrieval              |    Passed |
| Groundedness           |    Passed |
| Tool-use               |    Passed |
| Tool-reliability       |    Passed |
| Privacy                |    Passed |
| Conversation           |    Passed |
| Prompt-security        |    Passed |
| Multi-source-grounding |    Passed |
| Source-conflict        |    Passed |
| Abstention             |    Passed |
| **Total**              | **22/22** |

## Bug diary

### 1. Superseded policy occasionally outranked the current one in retrieval

`02-returns-policy-legacy.md` mentions "60 days" and "free return label". For the query "how long to return a backpack", raw TF-IDF similarity could place the legacy policy competitively close to `01-returns-policy-current.md`.

**Root cause:** TF-IDF has no built-in concept of document authority.

**Fix:** Added a status-based score multiplier in `Retriever.search()`:

* active = 1.0
* superseded = 0.55
* internal = 0.4

Regression test: `standard-return-window`, which verifies that legacy and internal sources are not treated as authoritative.

### 2. Cancelled/returned orders could show stale carrier and ETA fields

The first version of `OrderLookupTool.call()` passed through `carrier`, `tracking_number`, and `estimated_delivery` unconditionally. A cancelled order could therefore contain stale delivery information in the raw data.

**Fix:** `tools.py` now computes:

```python
show_delivery_fields = status not in ("cancelled", "returned")
```

and removes those delivery fields before the result reaches the model.

Regression test: `cancelled-order-stale-eta`.

### 3. Markdown chunking dropped the document's opening section

The first chunker only split on `## ` headings. Content between the YAML front matter and the first `##` heading could therefore be discarded.

For short documents with no subheadings, this could mean the entire document body was left unindexed.

**Fix:** `_chunk_markdown()` now explicitly captures the leading section as its own chunk before splitting the remaining content.

Regression testing verified that `12-breeze-tumbler-product-card.md` is retrievable for the relevant product-care conflict query.

## Known limitations / what I'd improve before production

* **TF-IDF retrieval is lexical, not semantic.** A real embedding model or hybrid BM25 + embedding reranker would handle paraphrases better than keyword-based similarity.

* **Concept assertions in the evaluation suite are keyword-overlap heuristics.** They can pass when the expected keywords appear in an incorrect context or fail when a correct answer is phrased unusually. Exact `must_include`, `must_not_include`, `sources`, `handoff`, and `tool_arguments` checks are more reliable.

* **No conversation-length or topic-drift guardrails.** A very long or wandering session could dilute the model's context. A production version could summarize or window older turns.

* **Single LLM provider.** There is currently no automatic provider fallback.

* **No retry/backoff on API errors.** A production implementation should handle rate limits and transient API failures gracefully.

* **The `<CONTROL>` block is a prompt convention rather than a hard schema.** It is parsed with a regex. A production version could replace this with a stricter structured-output approach.

## AI coding tools used

AI-assisted development was used during the project for architecture exploration, implementation assistance, debugging, retrieval logic, tool allowlisting, prompt design, and evaluation-harness development.

One example of an AI-generated suggestion that required correction was the first version of the markdown chunker. It only split content using `##` headings and therefore silently dropped content appearing before the first heading. This was discovered during testing of knowledge-base coverage and corrected by explicitly indexing the document's leading section.

## Demo

The demo video is included with the project submission as:

`FINAL DEMOO.mp4`

The demonstration covers the main reliability features of the agent, including:
https://drive.google.com/file/d/1Eio1SoP30YYtEjAJiBWS5EoNMTCMbEpY/view?usp=sharing


* Knowledge-base questions with citations
* Order lookup through the tool
* Multi-turn conversation
* Prompt-injection refusal / handoff behavior
* Evaluation suite execution
* Final evaluation result of **22/22 passed**

## Repository contents

```text
.
├── README.md
├── .env.example
├── requirements.txt
├── knowledge-base/        (14 policy/product markdown files)
├── data/
│   ├── orders.json
│   └── orders-data-dictionary.md
├── evaluation/
│   ├── visible-cases.json
│   ├── custom-cases.json  (7 original cases)
│   └── run_eval.py
├── src/
│   ├── retrieval.py
│   ├── tools.py
│   ├── agent.py
│   ├── logging_utils.py
│   └── cli.py
└── logs/                  (created at runtime, gitignored)
```
