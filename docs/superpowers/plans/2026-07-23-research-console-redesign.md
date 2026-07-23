# Research Console Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the existing Vue valuation UI into a source-aware Research Console that exposes model-selection evidence and remains fully usable for estimation.

**Architecture:** Keep the existing API and App-level data loading. Add one focused `ResearchOverview` component for v3 evidence rendering, make it the default view, and replace the global visual tokens/layout in `main.css`; existing form, assistant, result, and history components keep their behavior and receive only compatible class styling.

**Tech Stack:** Vue 3 script setup, Vite, Chart.js, native CSS, existing API client.

---

### Task 1: Add the research evidence surface

**Files:**
- Create: `frontend/src/components/ResearchOverview.vue`
- Modify: `frontend/src/App.vue`

- [ ] **Step 1: Render release identity and scope-separated metrics**

Use the model card fields `model_version`, `model_type`, `quality_gate`, `sample_count`, `data_source.source_id`, `cv_selection.winner_cv`, `independent_holdout.metrics`, and `split`. Never fall back from independent-test metrics to CV metrics when the v3 object exists.

- [ ] **Step 2: Render the top-five CV leaderboard**

Read `card.leaderboard.candidates`, sort by `cv.r2_mean` descending, show candidate name, model type, CV R2, CV 10% accuracy, and CV RMSE. Mark `card.leaderboard.winner` or `card.cv_selection.winner` as the released candidate.

- [ ] **Step 3: Render switchable error groups and technical contract**

Use tabs for `error_analysis.price_quartiles.groups`, `error_analysis.model_family_frequency.groups`, and `error_analysis.full_model_seen_status.groups`. Render only group summaries, not raw frequency dictionaries. Render feature descriptions as the technical input contract and show provenance/limitations with explicit empty states.

- [ ] **Step 4: Wire the view into App**

Add a `research` navigation item, make it the initial active view, render `ResearchOverview` for that view, and leave valuation/assistant/history routes intact.

- [ ] **Step 5: Build the frontend**

Run `npm run build` from `frontend`; expected output is a successful Vite production build.

### Task 2: Establish the research-console visual system

**Files:**
- Modify: `frontend/src/assets/base.css`
- Modify: `frontend/src/assets/main.css`

- [ ] **Step 1: Replace the green-only palette**

Set paper canvas, warm surface, ink, signal orange, cold blue, success teal, warning red, and line tokens. Keep contrast readable and avoid gradients and oversized decorative blobs.

- [ ] **Step 2: Style the new evidence layout**

Add stable grid tracks for release identity, metric split, leaderboard, error groups, technical contract, and provenance. Use small radii, thin rules, mono labels, and a restrained title treatment.

- [ ] **Step 3: Make all existing workflows consistent**

Restyle panels, forms, result states, assistant, history, loading, alert, focus, and footer selectors against the new tokens without changing their DOM contracts.

- [ ] **Step 4: Add responsive and reduced-motion rules**

At widths below 960px collapse the two-column workspace; below 720px turn the rail into horizontal scrolling tabs, stack metric cells, and preserve table scrolling. Keep text inside its containers and disable nonessential animation under `prefers-reduced-motion`.

### Task 3: Verify and browser-smoke the redesign

**Files:**
- Modify: `frontend/src/api.test.js` only if an API display contract needs a regression assertion.

- [ ] **Step 1: Run frontend tests and build**

Run `npm test -- --run` if a test script exists, otherwise run `npm run build` and `npm run dev -- --host 127.0.0.1 --port 5173` for browser verification.

- [ ] **Step 2: Start the backend and frontend**

Run the backend on `127.0.0.1:8000` and frontend on `127.0.0.1:5173`; confirm `/api/model-card` responds and the page loads without console errors.

- [ ] **Step 3: Check desktop and mobile browser states**

Use agent-browser to inspect the Research default view at desktop and narrow mobile widths, click Estimate and History navigation, switch error-analysis tabs, and confirm no overlapping text or blank evidence panels.

- [ ] **Step 4: Commit the frontend task**

Run `git diff --check`, `npm run build`, and commit the frontend/spec/plan changes with `feat: redesign research console`.
