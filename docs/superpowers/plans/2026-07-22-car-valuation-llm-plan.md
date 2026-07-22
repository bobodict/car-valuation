# Domain LLM Assistant Implementation Plan

> **For agentic workers:** Execute this plan task-by-task with tests first. The local repository has no requirement for a live provider during automated verification.

**Goal:** Add a domain LLM assistant that retrieves auditable car-market knowledge and calls the existing valuation model as a structured tool.

**Architecture:** A provider-neutral OpenAI-compatible client sends chat messages and one `estimate_vehicle` tool schema. A lexical knowledge retriever loads local JSON records and returns source ids. The assistant service executes validated tool arguments through the existing model adapter, then asks the LLM to produce a cited answer. Missing provider configuration returns an explicit 503 instead of a fake answer.

**Tech Stack:** FastAPI, Pydantic v2, Python standard-library HTTP client, Vue 3, local JSON knowledge base.

---

### Task 1: Add failing LLM tests

**Files:**
- Create: `car-valuation-submit/backend/tests/test_llm_contract.py`

- [ ] Test keyword retrieval returns source records.
- [ ] Test an unconfigured client raises `LLMNotConfiguredError`.
- [ ] Test a tool-call response validates vehicle arguments, calls the estimator, and returns citations.
- [ ] Run the backend test suite and verify these tests fail because the LLM modules and schemas do not exist.

### Task 2: Add LLM configuration, schemas, and knowledge retrieval

**Files:**
- Modify: `car-valuation-submit/backend/config.py`
- Modify: `car-valuation-submit/backend/schemas.py`
- Create: `car-valuation-submit/backend/services/knowledge_service.py`
- Create: `car-valuation-submit/backend/data/knowledge_base.json`

- [ ] Add provider URL, key, model, timeout, and knowledge path settings.
- [ ] Add assistant request/response, citation, and optional estimate schemas.
- [ ] Implement deterministic keyword retrieval with bounded results and source ids.
- [ ] Run retrieval and schema tests.

### Task 3: Implement provider client and assistant tool loop

**Files:**
- Create: `car-valuation-submit/backend/services/llm_client.py`
- Create: `car-valuation-submit/backend/services/assistant_service.py`
- Test: `car-valuation-submit/backend/tests/test_llm_contract.py`

- [ ] Implement standard-library HTTP calls to `/chat/completions` with injectable transport for tests.
- [ ] Define the `estimate_vehicle` tool schema.
- [ ] Validate tool arguments with `PredictRequest`, call `call_model_api`, append the tool result, and request the final cited answer.
- [ ] Return explicit disabled/provider errors and never invent an answer when the provider is unavailable.
- [ ] Run the complete backend suite.

### Task 4: Expose assistant API and configuration docs

**Files:**
- Modify: `car-valuation-submit/backend/main.py`
- Modify: `car-valuation-submit/backend/.env.example`
- Modify: `car-valuation-submit/backend/README.md`

- [ ] Add `POST /api/assistant` with 503 for disabled configuration and 502 for provider failures.
- [ ] Document provider setup, tool flow, knowledge citations, and offline behavior.
- [ ] Verify OpenAPI includes the new request and response contracts.

### Task 5: Add the assistant frontend surface

**Files:**
- Modify: `car-valuation-submit/frontend/src/App.vue`
- Modify: `car-valuation-submit/frontend/Readme.md`

- [ ] Add an assistant navigation page with message history, loading/error states, estimate output, and citations.
- [ ] Keep the assistant visibly separate from numeric model metrics and show disabled state when no provider is configured.
- [ ] Build the production frontend.

### Task 6: Final verification and push

**Files:**
- Verify only.

- [ ] Run backend tests, Python compilation, offline-disabled assistant behavior, simulated tool-call flow, and frontend build.
- [ ] Inspect staged files for secrets and generated output.
- [ ] Commit and push the LLM phase to `origin/main`.
