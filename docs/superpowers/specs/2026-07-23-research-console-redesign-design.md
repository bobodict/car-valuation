# Research Console Redesign

## Design Read

This is a research console for graduate-admissions reviewers and technical users. It should feel like a source-aware model audit notebook: quiet, precise, and readable under repeated inspection. The visual language uses warm paper, ink, signal orange, and cold blue rather than a generic SaaS gradient.

## Goals

- Make model evidence the first-class product surface, not a secondary chart.
- Keep development CV selection and the recorded independent holdout visually and semantically separate.
- Show the released model identity, candidate leaderboard, error groups, technical input contract, provenance, and limitations from the existing model-card API.
- Keep valuation, assistant, and history workflows available without duplicating backend logic.
- Preserve keyboard access, reduced-motion behavior, and a usable narrow mobile layout.

## Layout

```text
top bar: brand | API state | model version
--------------------------------------------------------------
rail: Research / Estimate / Assistant / History
main:
  research header + release identity
  metric split: development CV | recorded test
  candidate leaderboard       error analysis segments
  technical input contract    provenance + limitations
```

The Research view is the default route. A compact release identity block carries the model version, CatBoost family, quality gate, dataset rows, and source id. The metric split uses separate headers and colors, with `development_cv_only` and `recorded_test` scope labels rendered as data, not as decoration.

## Data Flow

The frontend continues to call `GET /api/model-card`, `GET /api/model-health`, `GET /api/metrics`, `GET /api/history`, and the existing prediction/assistant endpoints. `ResearchOverview.vue` consumes `card.cv_selection`, `card.independent_holdout`, `card.leaderboard.candidates`, `card.error_analysis`, `card.feature_descriptions`, and `card.data_source`. Missing optional v3 evidence renders a deliberate empty state so legacy cards remain compatible.

## Interaction

- Research segment tabs switch between price quartiles, model-family frequency, and seen/unseen model status without a route change.
- The leaderboard shows the top five CV candidates and highlights the published winner.
- The estimate view keeps its form and output panel, but the evidence link points back to the Research view.
- On mobile, the rail becomes a horizontal tab row and dense tables become scrollable regions with stable column widths.

## Error and Accessibility States

- API boot errors retain retry action and `role=alert`.
- Empty optional evidence explains that the publication is legacy or incomplete, rather than showing fabricated zeros.
- Quality-gate state uses text plus color and remains readable without color perception.
- All controls have visible focus states; charts and tables have accessible labels; reduced motion disables decorative transitions.

## Visual Tokens

- Canvas: `#f4f1eb`; surfaces: `#fffdf8`; ink: `#1f2528`.
- Signal orange: `#e3633b`; cold blue: `#2e647d`; success teal: `#1f806b`; muted line: `#d9d4ca`.
- Display typography: Georgia for the research title; body: system sans; technical labels and metrics: IBM Plex Mono fallback.
- Radius stays small (6px) and panels remain framed only where they contain a discrete tool or repeated evidence item.
