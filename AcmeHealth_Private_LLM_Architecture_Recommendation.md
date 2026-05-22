# Acme Health Private LLM -- Architecture Recommendation

**Audience:** Acme Health IT leadership, Migration / Data team
**Decision needed:** Approve the reference architecture and infra spend so
the team can stand up the first private-LLM workload (UKG -> Workday
migration) and reuse the same platform for clinical/admin assistants
later.

## TL;DR

- Run **Ollama** as the model server on a single GPU host (on-prem) or a
  single-tenant cloud VM (BAA in place). One stack, one budget line.
- Start with **`qwen2.5:7b-instruct`** as the default and
  **`llama3.1:8b-instruct`** as the alternate. Both are open-weight,
  permissively licensed, run on 16-24 GB VRAM, and handle the structured
  mapping prompts we need for the migration.
- Use **LangChain** for orchestration, **ChromaDB** for the eventual RAG
  index, **FAISS** as a lightweight fallback for prototypes.
- Put a **Streamlit** review surface in front of every workflow until
  trust is established.
- All inputs that touch PHI flow through a thin redaction layer; nothing
  leaves the VPC.

## 1. Why private, not API

| Concern                  | Public API LLM            | Private LLM (Ollama)              |
|--------------------------|---------------------------|-----------------------------------|
| HIPAA BAA                | Vendor-by-vendor, narrow  | We own the boundary.              |
| PHI in prompts/logs      | Implicit egress           | Stays on disk we control.         |
| Cost at scale            | Per-token, grows with use | Fixed GPU cost.                   |
| Determinism / pinning    | Vendor can swap models    | We pin the weights file.          |
| Latency (US-South region)| 300-1500 ms               | 50-200 ms on local GPU.           |
| Offline ops              | Requires connectivity     | Works in an air-gapped clinic.    |

The public-API option only really wins on raw capability for the largest
frontier models, and the UKG -> Workday workload doesn't need that.

## 2. Reference architecture

```
            +-------------------------------------------------+
            |               Acme Health VPC / on-prem              |
            |                                                 |
   UKG --+  |  +-------------+    +-------------+             |
   CSV   +-->|  ETL service  |--->|  SQLite /   |             |
   drop  |  |  (Pandas)      |    |  Postgres   |             |
         |  |                |    |  staging    |             |
         |  +-------+--------+    +------+------+             |
         |          |                    |                    |
         |          v                    v                    |
         |  +---------------+   +--------+---------+          |
         |  |  Redaction    |   |  Vector store    |          |
         |  |  layer        |   |  (ChromaDB)      |          |
         |  +-------+-------+   +--------+---------+          |
         |          |                    |                    |
         |          v                    v                    |
         |        +-----------------------------+             |
         |        |  LangChain orchestrator     |             |
         |        +--------------+--------------+             |
         |                       |                            |
         |                       v                            |
         |              +---------------+                     |
         |              |  Ollama       |                     |
         |              |  (GPU host)   |                     |
         |              +-------+-------+                     |
         |                      |                             |
         |   +------------------+----------------+            |
         |   |                                   |            |
         |   v                                   v            |
         |  Streamlit                       Workday EIB /     |
         |  review surface                  API loader        |
         +-------------------------------------------------+
```

## 3. Model recommendation

### Default: `qwen2.5:7b-instruct`
- Strong instruction following, especially for JSON-constrained outputs
  (we lean on this for the mapping prompt).
- Apache 2.0 license; safe for commercial use.
- ~5 GB on disk in Q4 quantization, ~14 GB VRAM at FP16.

### Alternate: `llama3.1:8b-instruct`
- Slightly stronger on long-context reasoning.
- Llama 3 community license -- review terms before commercial scale-out.

### When we'd upgrade
- Move to **`qwen2.5:14b`** or **`llama3.1:70b`** once we add clinical
  Q&A workloads. 70B requires 2x 48GB GPUs or sharding.
- Domain-tune on Acme Health's own SOPs and historic ticket data only after
  the migration use case is stable.

### What we wouldn't pick
- `tinyllama` / sub-3B models -- too brittle on structured mapping.
- A code-tuned model (e.g. `codellama`) -- wrong specialty.

## 4. Infrastructure sizing

### Phase 1 -- migration only
- **1x GPU host**: RTX A5000 (24 GB) or RTX 4090 (24 GB), 64 GB RAM,
  2 TB NVMe.
- Ollama + ChromaDB + Postgres on the same box for simplicity.
- Cost: ~$6-10K capex on-prem or ~$1-1.5K/mo as a single-tenant cloud VM.

### Phase 2 -- migration + admin assistants
- **Add a second GPU host** for HA and to separate inference from
  background re-indexing.
- Move Postgres + Chroma to dedicated instances.

### Phase 3 -- clinical workloads
- **Add a 48-80 GB VRAM tier** (L40S or H100 partition) for larger
  models and longer contexts.
- Introduce a model gateway (e.g. LiteLLM) so apps don't need to know
  which backend they're hitting.

## 5. Security and HIPAA posture

| Control                              | How we implement it                          |
|--------------------------------------|----------------------------------------------|
| Encryption at rest                   | LUKS / EBS-CMK on the GPU host and storage. |
| Encryption in transit                | TLS on every internal hop, even VPC-local.  |
| Access control                       | SSO + RBAC at the Streamlit / API layer.    |
| Prompt audit                         | `audit.llm_prompts` table -- prompt, model, response, user. |
| Egress                               | NACLs deny all egress from the GPU subnet.  |
| Redaction                            | Allow-list of fields before any prompt send. |
| Model provenance                     | SHA-256 of weights pinned in config.        |
| Patch cadence                        | Quarterly model review; security patches monthly. |

## 6. Decision matrix -- RAG vs fine-tune vs prompt-only

| Use case                            | Approach              | Rationale                              |
|-------------------------------------|-----------------------|----------------------------------------|
| UKG -> Workday mapping              | Prompt-only, few-shot | Vocabulary is small and stable.        |
| Policy Q&A (HR handbook, etc.)      | RAG over Chroma       | Content changes; need citations.       |
| Clinical decision support           | Not in scope yet      | Liability + regulatory bar.            |
| Long-tail intake summarization      | RAG + light fine-tune | Style consistency benefits from tuning.|

## 7. Build vs buy on the LLM platform layer

We considered Healthie / Datavant / Hippocratic-style vendor stacks.
Recommendation: **buy nothing yet**. Open-source Ollama + LangChain +
Chroma gives us 90% of what a managed stack would, with the boundary
fully under Acme Health's control. Revisit the buy question when we have
three+ production workloads and on-call burden becomes the bottleneck.

## 8. Risks and mitigations

| Risk                                          | Mitigation                                     |
|-----------------------------------------------|------------------------------------------------|
| GPU host single point of failure              | Replicate to a second host in Phase 2.         |
| Model output hallucination on edge cases      | Confidence threshold + human-in-the-loop.      |
| Open-source license drift (Llama)             | Prefer Apache-licensed alternatives (Qwen).    |
| Cost creep                                    | Cap context length per workload; cache prompts.|
| Skills concentration                          | Pair-program; write runbooks in `runbooks/`.   |

## 9. Recommended next 30 days

1. **Week 1:** Procure GPU host (or stand up cloud VM); install Ollama;
   pull `qwen2.5:7b-instruct`.
2. **Week 2:** Migration take-home is the proving ground -- Rithika's
   submission feeds the first prod pipeline.
3. **Week 3:** Wire the prompt audit table; security review.
4. **Week 4:** Run a parallel test load (UKG -> Workday sandbox) and
   compare to the manual mapping.
