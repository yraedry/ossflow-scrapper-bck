# Split de Repos — Cierre del Monorepo (2026-04-29)

Este monorepo se ha dividido en 4 repos independientes:

- [`ossflow-core`](https://github.com/yraedry/ossflow-core) — infraestructura compartida (imagen base + service kit)
- [`ossflow-platform`](https://github.com/yraedry/ossflow-platform) — backend orquestado (api, splitter, subtitle, dubbing, telegram)
- [`ossflow-scrapper`](https://github.com/yraedry/ossflow-scrapper) — scraping de proveedores (BJJ Fanatics + futuros)
- [`ossflow-studio`](https://github.com/yraedry/ossflow-studio) — frontend SPA React

Ver el spec completo en `docs/superpowers/specs/2026-04-29-ossflow-architecture-refactor-design.md`
y el plan de ejecución en `docs/superpowers/plans/2026-04-29-ossflow-repo-split.md`.

Tag de seguridad pre-refactor: `v-pre-refactor`.
