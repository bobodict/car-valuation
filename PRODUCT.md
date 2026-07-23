# Product

## Register

product

## Users

Technical reviewers and technical users who need to inspect, test, and explain a used-car valuation model. The primary context is repeated review of model evidence, with occasional interactive valuation runs and traceable assistance.

## Product Purpose

Car Valuation is a source-aware used-car valuation research console. It combines a reproducible training pipeline, a versioned model publication, an auditable model card, and a Vue/FastAPI application. Success means a reviewer can distinguish model-selection evidence from the independent holdout, understand the input contract and dataset limits, and run an estimate without mistaking an experimental artifact for a production guarantee.

## Brand Personality

Quiet, precise, accountable. The interface should feel like a readable research notebook: confident about evidence, explicit about limitations, and practical during repeated inspection.

## Anti-references

Avoid generic SaaS dashboards, promotional landing-page composition, gradient-heavy visual systems, decorative chart clutter, unexplained confidence claims, and interfaces that hide data provenance behind a prediction button.

## Design Principles

- Evidence before persuasion: surface source, split, metrics, and limitations beside the result.
- Separate scopes honestly: development CV selects candidates; the recorded holdout evaluates the release.
- Keep the research workflow primary while preserving estimation, assistant, and history actions.
- Make uncertainty legible through explicit quality-gate states and empty evidence states.
- Prefer durable, inspectable interactions over decorative motion or opaque automation.

## Accessibility & Inclusion

Target WCAG 2.1 AA practices for the application surface. Preserve keyboard focus, semantic labels, readable contrast, and text equivalents for quality states. Do not rely on color alone for pass/fail meaning, keep tables usable on narrow screens, and honor reduced-motion preferences.
