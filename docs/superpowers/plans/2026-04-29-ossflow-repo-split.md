# Plan 1 — Split de Repos OSSFlow

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dividir el monorepo actual `ossflow-scrapper` en cuatro repositorios independientes (`ossflow-core`, `ossflow-platform`, `ossflow-scrapper`, `ossflow-studio`), renombrar todos los identificadores `bjj_*` → `ossflow_*`, y dejar todos los servicios arrancando con sus tests pasando. **No se refactoriza la arquitectura interna en este plan** — solo se mueve y renombra código.

**Architecture:** Big bang en cuatro repos. Cada repo se prepara secuencialmente: crear repo vacío → copiar código → renombrar paquetes Python → renombrar imágenes y servicios Docker → adaptar Dockerfiles y compose → verificar arranque y tests. El `docker-compose.yml` final vive en `ossflow-platform` y referencia el paquete `ossflow-service-kit` instalado vía `git+https://...@v1.0.0`.

**Tech Stack:** Python 3.11+, FastAPI, SQLAlchemy 2.0, Pydantic v2, pytest, Docker Compose, React 18 + Vite (frontend, sin cambios internos), GitHub para versionado.

---

## Convenciones del plan

- **Idioma:** todos los comentarios y docstrings nuevos/modificados en castellano. El código existente se renombra sin reescribir comentarios salvo cuando un comentario menciona explícitamente `bjj_*`.
- **Working dir referencia:** `C:/proyectos/python/ossflow-scrapper` es el monorepo origen. Los nuevos repos se crearán como hermanos: `C:/proyectos/python/ossflow-core`, `C:/proyectos/python/ossflow-platform`, `C:/proyectos/python/ossflow-scrapper-new`, `C:/proyectos/python/ossflow-studio`. Al final el monorepo origen se archiva.
- **Nombres definitivos:**
  - Imagen base Docker: `bjj-base:latest` → `ossflow-base:latest`
  - Paquete Python compartido: `bjj_service_kit` → `ossflow_service_kit` (importable con guión bajo, `pip install ossflow-service-kit`)
  - Network compose: `bjj_net` → `ossflow_net`
  - Variable env: `BJJ_DB_PATH` → `OSSFLOW_DB_PATH` (mantener compatibilidad: leer ambas en código durante una transición controlada por una sola tarea)
  - Volumen DB: `bjj-db` → `ossflow-db`
  - Tag Git inicial del kit: `v0.1.0`
- **Sustitución del nombre de usuario GitHub:** las URLs `git+https://github.com/<usuario>/...` usan el placeholder `<GITHUB_USER>` que el ejecutor debe sustituir por el handle real de Adrián antes del push (única excepción a "no placeholders" — se documenta aquí en una sección, y cada Dockerfile/pyproject afectado lleva una nota visible).
- **Tag de seguridad:** antes de empezar, taggear `v-pre-refactor` en el repo origen.
- **Frecuencia de commits:** un commit por tarea completada como mínimo. Push a remoto al final de cada repo (o más frecuente si se prefiere).

---

## Mapeo de Archivos y Repositorios

### `ossflow-core`

```
ossflow-core/
├── ossflow-base/                    ← copia de bjj-base/
│   └── Dockerfile                   ← FROM... + comentarios actualizados
├── ossflow_service_kit/             ← copia de bjj_service_kit/
│   ├── __init__.py
│   ├── app_factory.py
│   ├── events.py
│   ├── log_bridge.py
│   ├── runner.py
│   ├── schemas.py
│   ├── pyproject.toml               ← name: ossflow-service-kit
│   ├── db/
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   ├── session.py
│   │   ├── models.py
│   │   └── migrations/...
│   └── tests/
├── README.md
└── .gitignore
```

### `ossflow-platform`

```
ossflow-platform/
├── ossflow-api/                     ← copia de processor-api/
├── ossflow-splitter/                ← copia de chapter-splitter/
├── ossflow-subtitle/                ← copia de subtitle-generator/
├── ossflow-dubbing/                 ← copia de dubbing-generator/
├── ossflow-telegram/                ← copia de telegram-fetcher/
├── docker-compose.yml               ← copiado y reescrito
├── docker-compose.override.yml      ← copiado y reescrito
├── ollama/                          ← copia de ollama/
├── models/                          ← copia de models/
├── .env.example
├── README.md
└── .gitignore
```

### `ossflow-scrapper` (nuevo, NO el monorepo origen)

> **Nota de naming:** el repositorio monorepo actual se llama `ossflow-scrapper` por accidente histórico. El nuevo repo dedicado al scrapper también se llama `ossflow-scrapper`. Para evitar conflicto durante el split, el nuevo repo se prepara en `C:/proyectos/python/ossflow-scrapper-new/` y al final, cuando el monorepo origen se archive, se renombra el directorio local a `ossflow-scrapper/`. En GitHub el monorepo origen se renombra antes del push (ej. `ossflow-monorepo-archive`) para liberar el nombre.

```
ossflow-scrapper/
├── app.py                           ← extraído de chapter-splitter/app_oracle.py
├── Dockerfile                       ← extraído de chapter-splitter/Dockerfile.oracle
├── providers/                       ← extraído de chapter-splitter/<oracle providers>
├── tests/
├── requirements.txt
├── pyproject.toml
├── README.md
└── .gitignore
```

### `ossflow-studio`

```
ossflow-studio/
├── ossflow-frontend/                ← copia de processor-frontend/
└── README.md
```

---

## Tareas

### Fase 0: Preparación

#### Task 1: Tag de seguridad y backup

**Files:**
- Modify: estado de git en `C:/proyectos/python/ossflow-scrapper`
- Create: backup local de la BD SQLite en `C:/proyectos/python/.backups/bjj-pre-refactor.db`

- [ ] **Step 1: Verificar estado limpio del repo**

Run:
```bash
cd C:/proyectos/python/ossflow-scrapper
git status
```

Expected: `nothing to commit, working tree clean`. Si hay cambios sin commitear, parar y resolver antes de continuar.

- [ ] **Step 2: Crear tag de seguridad**

Run:
```bash
cd C:/proyectos/python/ossflow-scrapper
git tag v-pre-refactor
git push origin v-pre-refactor
```

Expected: tag pusheado al remoto sin error.

- [ ] **Step 3: Backup de la BD del contenedor**

Run:
```bash
mkdir -p C:/proyectos/python/.backups
docker compose -f C:/proyectos/python/ossflow-scrapper/docker-compose.yml cp processor-api:/data/db/bjj.db C:/proyectos/python/.backups/bjj-pre-refactor.db
```

Expected: archivo `C:/proyectos/python/.backups/bjj-pre-refactor.db` creado con tamaño > 0. Si los contenedores no están arriba, levantar primero con `docker compose up -d processor-api` o copiar manualmente desde el volumen Docker.

- [ ] **Step 4: Detener todos los contenedores**

Run:
```bash
docker compose -f C:/proyectos/python/ossflow-scrapper/docker-compose.yml down
```

Expected: todos los contenedores detenidos. Esto evita que durante el split haya código viejo escribiendo en la BD.

#### Task 2: Crear cuatro repositorios vacíos en GitHub

**Files:**
- Create: 4 repos en GitHub (acción manual en navegador o `gh repo create`)

- [ ] **Step 1: Crear repos en GitHub (privados)**

Run (requiere `gh` CLI autenticado):
```bash
gh repo create <GITHUB_USER>/ossflow-core --private --description "OSSFlow shared infrastructure: Docker base + Python service kit"
gh repo create <GITHUB_USER>/ossflow-platform --private --description "OSSFlow platform: API gateway, splitter, subtitle, dubbing, telegram"
gh repo create <GITHUB_USER>/ossflow-scrapper --private --description "OSSFlow scrapper: BJJ Fanatics + future providers"
gh repo create <GITHUB_USER>/ossflow-studio --private --description "OSSFlow Studio: React frontend SPA"
```

Expected: cada comando devuelve la URL del repo creado. Si el nombre `ossflow-scrapper` ya está ocupado por el monorepo, primero renombrar el monorepo en GitHub (`gh repo rename ossflow-monorepo-archive`).

- [ ] **Step 2: Verificar acceso a los 4 repos**

Run:
```bash
gh repo view <GITHUB_USER>/ossflow-core --json name,visibility
gh repo view <GITHUB_USER>/ossflow-platform --json name,visibility
gh repo view <GITHUB_USER>/ossflow-scrapper --json name,visibility
gh repo view <GITHUB_USER>/ossflow-studio --json name,visibility
```

Expected: cada uno devuelve JSON con el nombre y `"visibility": "PRIVATE"`.

---

### Fase 1: `ossflow-core`

#### Task 3: Crear estructura local de `ossflow-core`

**Files:**
- Create: `C:/proyectos/python/ossflow-core/`
- Create: `C:/proyectos/python/ossflow-core/.gitignore`
- Create: `C:/proyectos/python/ossflow-core/README.md`

- [ ] **Step 1: Crear directorio del repo**

Run:
```bash
mkdir -p C:/proyectos/python/ossflow-core
cd C:/proyectos/python/ossflow-core
git init
git remote add origin https://github.com/<GITHUB_USER>/ossflow-core.git
```

Expected: directorio creado, repo inicializado, remoto configurado.

- [ ] **Step 2: Copiar `bjj-base/` → `ossflow-base/`**

Run (PowerShell):
```powershell
Copy-Item -Recurse C:/proyectos/python/ossflow-scrapper/bjj-base C:/proyectos/python/ossflow-core/ossflow-base
```

Expected: el directorio `ossflow-core/ossflow-base/` contiene `Dockerfile` (y cualquier otro archivo que estuviera en `bjj-base/`).

- [ ] **Step 3: Copiar `bjj_service_kit/` → `ossflow_service_kit/`**

Run (PowerShell):
```powershell
Copy-Item -Recurse C:/proyectos/python/ossflow-scrapper/bjj_service_kit C:/proyectos/python/ossflow-core/ossflow_service_kit
Remove-Item -Recurse -Force C:/proyectos/python/ossflow-core/ossflow_service_kit/__pycache__ -ErrorAction SilentlyContinue
```

Expected: el paquete copiado, sin `__pycache__`.

- [ ] **Step 4: Crear `.gitignore`**

Create `C:/proyectos/python/ossflow-core/.gitignore`:

```
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.venv/
venv/
.env
.env.local
*.egg-info/
build/
dist/
.coverage
htmlcov/
.idea/
.vscode/
```

- [ ] **Step 5: Crear `README.md` mínimo**

Create `C:/proyectos/python/ossflow-core/README.md`:

```markdown
# ossflow-core

Infraestructura compartida del ecosistema OSSFlow.

## Contenido

- **`ossflow-base/`**: imagen Docker base con CUDA 12.4 + PyTorch + FastAPI. Se construye una sola vez (`docker build -t ossflow-base:latest -f ossflow-base/Dockerfile .`) y la consumen los servicios GPU de `ossflow-platform`.
- **`ossflow_service_kit/`**: paquete Python compartido (`ossflow-service-kit` en PyPI). Provee `app_factory`, ring buffer de logs, eventos SSE, runner de jobs y la capa de base de datos común (SQLAlchemy 2.0 + Alembic).

## Distribución

`ossflow_service_kit` se publica vía Git tags. Los servicios consumidores instalan con:

```toml
dependencies = [
    "ossflow-service-kit @ git+https://github.com/<GITHUB_USER>/ossflow-core@v0.1.0#subdirectory=ossflow_service_kit",
]
```
```

- [ ] **Step 6: Commit inicial**

Run:
```bash
cd C:/proyectos/python/ossflow-core
git add .
git commit -m "chore: importar código original desde monorepo (sin renombrar)"
```

Expected: commit creado. Aún `bjj_*` por dentro — el rename va en la siguiente tarea.

#### Task 4: Renombrar `bjj_service_kit` → `ossflow_service_kit` en el código

**Files:**
- Modify: `C:/proyectos/python/ossflow-core/ossflow_service_kit/pyproject.toml`
- Modify: todos los `*.py` dentro de `ossflow_service_kit/` que mencionen `bjj_service_kit`
- Modify: `C:/proyectos/python/ossflow-core/ossflow_service_kit/db/migrations/env.py`

- [ ] **Step 1: Reescribir `pyproject.toml`**

Replace `C:/proyectos/python/ossflow-core/ossflow_service_kit/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "ossflow-service-kit"
version = "0.1.0"
description = "Kit FastAPI compartido para los servicios OSSFlow"
requires-python = ">=3.10"
dependencies = [
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "sse-starlette>=2.0",
    "pydantic>=2.5",
    "sqlalchemy>=2.0",
    "alembic>=1.13",
]

[project.optional-dependencies]
test = [
    "pytest>=7.4",
    "pytest-asyncio>=0.23",
    "httpx>=0.27",
]

# Layout plano: el directorio ES el paquete. Mapeamos el nombre
# ossflow_service_kit a "." para que setuptools no salga del project root.
[tool.setuptools]
packages = [
    "ossflow_service_kit",
    "ossflow_service_kit.db",
    "ossflow_service_kit.db.migrations",
    "ossflow_service_kit.db.migrations.versions",
]

[tool.setuptools.package-dir]
ossflow_service_kit = "."
"ossflow_service_kit.db" = "db"
"ossflow_service_kit.db.migrations" = "db/migrations"
"ossflow_service_kit.db.migrations.versions" = "db/migrations/versions"
```

- [ ] **Step 2: Reemplazar todas las menciones a `bjj_service_kit` dentro del paquete**

Run (PowerShell):
```powershell
$files = Get-ChildItem -Path C:/proyectos/python/ossflow-core/ossflow_service_kit -Recurse -Include *.py,*.cfg,*.ini,*.toml,*.txt,*.md
foreach ($f in $files) {
  (Get-Content $f.FullName) -replace 'bjj_service_kit', 'ossflow_service_kit' -replace 'bjj-service-kit', 'ossflow-service-kit' | Set-Content $f.FullName
}
```

Expected: todas las menciones reescritas. `pyproject.toml` ya está reemplazado en el step 1; el grep en el step 4 confirmará que no queda ninguna.

- [ ] **Step 3: Comprobar que Alembic (`db/migrations/env.py`) sigue siendo correcto**

Read `C:/proyectos/python/ossflow-core/ossflow_service_kit/db/migrations/env.py` y verificar que cualquier `from bjj_service_kit...` quedó como `from ossflow_service_kit...` tras el step 2.

Expected: sin literales `bjj_service_kit` ni `bjj-service-kit`.

- [ ] **Step 4: Verificar con grep que no quedan referencias**

Run:
```bash
cd C:/proyectos/python/ossflow-core
grep -rn "bjj_service_kit\|bjj-service-kit" ossflow_service_kit
```

Expected: cero resultados.

- [ ] **Step 5: Instalar el paquete en un venv local y correr los tests**

Run:
```bash
cd C:/proyectos/python/ossflow-core/ossflow_service_kit
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[test]"
.venv/Scripts/python.exe -m pytest tests/ -v
```

Expected: instalación correcta y todos los tests pasando.

- [ ] **Step 6: Commit**

Run:
```bash
cd C:/proyectos/python/ossflow-core
git add .
git commit -m "refactor: renombrar bjj_service_kit -> ossflow_service_kit"
```

#### Task 5: Renombrar imagen Docker base `bjj-base` → `ossflow-base`

**Files:**
- Modify: `C:/proyectos/python/ossflow-core/ossflow-base/Dockerfile`

- [ ] **Step 1: Leer el Dockerfile actual**

Run:
```bash
cat C:/proyectos/python/ossflow-core/ossflow-base/Dockerfile | head -20
```

Anotar el `FROM` y cualquier comentario de cabecera.

- [ ] **Step 2: Sustituir `bjj-base` y `bjj_base` en el Dockerfile**

Run (PowerShell):
```powershell
$f = "C:/proyectos/python/ossflow-core/ossflow-base/Dockerfile"
(Get-Content $f) -replace 'bjj-base', 'ossflow-base' -replace 'bjj_base', 'ossflow_base' | Set-Content $f
```

- [ ] **Step 3: Verificar**

Run:
```bash
grep -n "bjj" C:/proyectos/python/ossflow-core/ossflow-base/Dockerfile
```

Expected: cero resultados.

- [ ] **Step 4: Construir la imagen base nueva**

Run:
```bash
cd C:/proyectos/python/ossflow-core
docker build -t ossflow-base:latest -f ossflow-base/Dockerfile .
```

Expected: build exitoso, imagen `ossflow-base:latest` listada en `docker images`.

- [ ] **Step 5: Verificar imagen disponible**

Run:
```bash
docker images ossflow-base
```

Expected: una entrada con tag `latest`.

- [ ] **Step 6: Commit**

Run:
```bash
cd C:/proyectos/python/ossflow-core
git add ossflow-base/Dockerfile
git commit -m "refactor: renombrar imagen bjj-base -> ossflow-base"
```

#### Task 6: Push de `ossflow-core` y crear tag `v0.1.0`

**Files:**
- Push remoto

- [ ] **Step 1: Push de la rama main**

Run:
```bash
cd C:/proyectos/python/ossflow-core
git branch -M main
git push -u origin main
```

Expected: rama `main` pusheada.

- [ ] **Step 2: Crear y pushear tag de versión**

Run:
```bash
cd C:/proyectos/python/ossflow-core
git tag v0.1.0
git push origin v0.1.0
```

Expected: tag visible en `gh release list` o navegando al repo en GitHub.

- [ ] **Step 3: Probar instalación remota del kit (smoke test)**

Run:
```bash
python -m venv C:/tmp/ossflow-kit-test
C:/tmp/ossflow-kit-test/Scripts/python.exe -m pip install "ossflow-service-kit @ git+https://github.com/<GITHUB_USER>/ossflow-core@v0.1.0#subdirectory=ossflow_service_kit"
C:/tmp/ossflow-kit-test/Scripts/python.exe -c "import ossflow_service_kit; print(ossflow_service_kit.__file__)"
```

Expected: el comando imprime una ruta dentro del venv `C:/tmp/ossflow-kit-test/...`. Si falla la instalación remota, el problema típicamente es el subdirectory path o que el repo es privado y falta autenticación — resolver antes de continuar.

- [ ] **Step 4: Limpiar venv de prueba**

Run (PowerShell):
```powershell
Remove-Item -Recurse -Force C:/tmp/ossflow-kit-test
```

---

### Fase 2: `ossflow-platform`

#### Task 7: Crear estructura local de `ossflow-platform`

**Files:**
- Create: `C:/proyectos/python/ossflow-platform/`
- Create: `.gitignore`, `README.md`

- [ ] **Step 1: Crear repo local**

Run:
```bash
mkdir -p C:/proyectos/python/ossflow-platform
cd C:/proyectos/python/ossflow-platform
git init
git remote add origin https://github.com/<GITHUB_USER>/ossflow-platform.git
```

- [ ] **Step 2: Copiar servicios renombrando carpetas**

Run (PowerShell):
```powershell
$src = "C:/proyectos/python/ossflow-scrapper"
$dst = "C:/proyectos/python/ossflow-platform"
Copy-Item -Recurse "$src/processor-api" "$dst/ossflow-api"
Copy-Item -Recurse "$src/chapter-splitter" "$dst/ossflow-splitter"
Copy-Item -Recurse "$src/subtitle-generator" "$dst/ossflow-subtitle"
Copy-Item -Recurse "$src/dubbing-generator" "$dst/ossflow-dubbing"
Copy-Item -Recurse "$src/telegram-fetcher" "$dst/ossflow-telegram"
Copy-Item -Recurse "$src/ollama" "$dst/ollama"
Copy-Item -Recurse "$src/models" "$dst/models" -ErrorAction SilentlyContinue
Copy-Item "$src/docker-compose.yml" "$dst/docker-compose.yml"
Copy-Item "$src/docker-compose.override.yml" "$dst/docker-compose.override.yml"
Copy-Item "$src/instructional_detail.yml" "$dst/instructional_detail.yml"
```

Expected: la estructura del repo platform tiene los 5 servicios renombrados + ollama + models + composes.

- [ ] **Step 3: Eliminar `__pycache__` y `.pytest_cache` heredados**

Run (PowerShell):
```powershell
Get-ChildItem -Path C:/proyectos/python/ossflow-platform -Recurse -Directory -Include __pycache__,.pytest_cache | Remove-Item -Recurse -Force
```

- [ ] **Step 4: Crear `.gitignore`**

Create `C:/proyectos/python/ossflow-platform/.gitignore`:

```
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.venv/
venv/
.env
.env.local
*.egg-info/
build/
dist/
.coverage
htmlcov/
.idea/
.vscode/
docs/superpowers/
```

- [ ] **Step 5: Crear `README.md` placeholder**

Create `C:/proyectos/python/ossflow-platform/README.md`:

```markdown
# ossflow-platform

Backend orquestado del ecosistema OSSFlow para procesamiento de instruccionales BJJ.

## Servicios

| Servicio | Puerto | Descripción |
|---|---|---|
| `ossflow-api` | 8000 | Gateway FastAPI: orquesta backends, gestiona pipelines y biblioteca |
| `ossflow-splitter` | 8001 | Fragmentación en capítulos (modo signal o timestamps) |
| `ossflow-subtitle` | 8002 | Generación de subtítulos con WhisperX |
| `ossflow-dubbing` | 8003 | Doblaje vía ElevenLabs |
| `ossflow-telegram` | 8004 | Descarga de media desde canales Telegram |

## Arranque

```bash
docker compose build
docker compose up -d
```

Requiere `ossflow-base:latest` construido previamente desde [`ossflow-core`](https://github.com/<GITHUB_USER>/ossflow-core).
```

- [ ] **Step 6: Eliminar Dockerfile.oracle del splitter (se va al repo scrapper)**

Run (PowerShell):
```powershell
Remove-Item C:/proyectos/python/ossflow-platform/ossflow-splitter/Dockerfile.oracle -ErrorAction SilentlyContinue
Remove-Item C:/proyectos/python/ossflow-platform/ossflow-splitter/app_oracle.py -ErrorAction SilentlyContinue
```

> Nota: `app_oracle.py` y `Dockerfile.oracle` se moverán al repo `ossflow-scrapper`. Aquí los borramos del platform para que el splitter quede solo con su responsabilidad real (modo signal). En la Fase 3 confirmaremos que los archivos van al destino correcto antes de borrar definitivamente del monorepo origen.

- [ ] **Step 7: Commit inicial**

Run:
```bash
cd C:/proyectos/python/ossflow-platform
git add .
git commit -m "chore: importar servicios desde monorepo (sin renombrar imports)"
```

#### Task 8: Renombrar imports y referencias `bjj_service_kit` → `ossflow_service_kit` en todos los servicios

**Files:**
- Modify: todos los `*.py` en `ossflow-api/`, `ossflow-splitter/`, `ossflow-subtitle/`, `ossflow-dubbing/`, `ossflow-telegram/`

- [ ] **Step 1: Sustituir en todos los archivos Python**

Run (PowerShell):
```powershell
$root = "C:/proyectos/python/ossflow-platform"
$files = Get-ChildItem -Path $root -Recurse -Include *.py
foreach ($f in $files) {
  (Get-Content $f.FullName) -replace 'bjj_service_kit', 'ossflow_service_kit' -replace 'bjj-service-kit', 'ossflow-service-kit' | Set-Content $f.FullName
}
```

- [ ] **Step 2: Verificar con grep**

Run:
```bash
cd C:/proyectos/python/ossflow-platform
grep -rn "bjj_service_kit\|bjj-service-kit" --include="*.py"
```

Expected: cero resultados.

- [ ] **Step 3: Commit**

Run:
```bash
cd C:/proyectos/python/ossflow-platform
git add -A
git commit -m "refactor: actualizar imports bjj_service_kit -> ossflow_service_kit"
```

#### Task 9: Actualizar Dockerfiles de cada servicio para consumir `ossflow-service-kit` desde Git

**Files:**
- Modify: `ossflow-api/Dockerfile`
- Modify: `ossflow-splitter/Dockerfile`
- Modify: `ossflow-subtitle/Dockerfile`
- Modify: `ossflow-dubbing/Dockerfile`
- Modify: `ossflow-telegram/Dockerfile`

> **Cambio conceptual:** los Dockerfiles ya no copian `bjj_service_kit/` desde el contexto de build (porque ya no está en este repo). Instalan el paquete vía `pip install "ossflow-service-kit @ git+https://github.com/<GITHUB_USER>/ossflow-core@v0.1.0#subdirectory=ossflow_service_kit"`.

- [ ] **Step 1: Modificar `ossflow-api/Dockerfile`**

Read `C:/proyectos/python/ossflow-platform/ossflow-api/Dockerfile` para identificar el bloque actual:

```dockerfile
COPY bjj_service_kit /opt/bjj_service_kit
RUN pip install --no-cache-dir /opt/bjj_service_kit
```

Replace ese bloque por:

```dockerfile
# ossflow-service-kit se instala desde GitHub (versionado por tag).
# Para usar una versión local en desarrollo, sustituir esta línea por
# COPY ../ossflow-core/ossflow_service_kit /opt/ossflow_service_kit
# y ajustar el contexto de build.
RUN pip install --no-cache-dir "ossflow-service-kit @ git+https://github.com/<GITHUB_USER>/ossflow-core@v0.1.0#subdirectory=ossflow_service_kit"
```

Y eliminar el comentario antiguo que decía `# Build context: ../python (repo root) so bjj_service_kit is visible.`

- [ ] **Step 2: Cambiar `FROM bjj-base:latest` → `FROM ossflow-base:latest` en cada Dockerfile que lo use**

Run (PowerShell):
```powershell
$dockerfiles = @(
  "C:/proyectos/python/ossflow-platform/ossflow-api/Dockerfile",
  "C:/proyectos/python/ossflow-platform/ossflow-splitter/Dockerfile",
  "C:/proyectos/python/ossflow-platform/ossflow-subtitle/Dockerfile",
  "C:/proyectos/python/ossflow-platform/ossflow-dubbing/Dockerfile",
  "C:/proyectos/python/ossflow-platform/ossflow-telegram/Dockerfile"
)
foreach ($d in $dockerfiles) {
  if (Test-Path $d) {
    (Get-Content $d) -replace 'bjj-base', 'ossflow-base' -replace 'bjj_base', 'ossflow_base' | Set-Content $d
  }
}
```

- [ ] **Step 3: Aplicar el patrón del Step 1 a los Dockerfiles restantes**

Para cada uno de los siguientes Dockerfiles, encontrar el patrón:

```dockerfile
COPY bjj_service_kit /app/bjj_service_kit
... && pip install -e /app/bjj_service_kit ...
```

(o la variante `COPY bjj_service_kit /opt/bjj_service_kit && pip install --no-cache-dir /opt/bjj_service_kit`)

Y reemplazarlo por:

```dockerfile
RUN pip install --no-cache-dir "ossflow-service-kit @ git+https://github.com/<GITHUB_USER>/ossflow-core@v0.1.0#subdirectory=ossflow_service_kit"
```

Aplicar a:
- `ossflow-splitter/Dockerfile` (modo signal — el único Dockerfile que queda en splitter tras el Step 6 de Task 7)
- `ossflow-subtitle/Dockerfile`
- `ossflow-dubbing/Dockerfile`
- `ossflow-telegram/Dockerfile`

> **Importante:** mantener cualquier `RUN pip install -r requirements.txt` que ya estuviera. Solo se reemplaza la instalación del kit.

- [ ] **Step 4: Verificar que ningún Dockerfile contiene `bjj_*` ni `COPY bjj_service_kit`**

Run:
```bash
cd C:/proyectos/python/ossflow-platform
grep -rn "bjj_service_kit\|bjj-base\|bjj_base\|bjj-service-kit" --include="Dockerfile*"
```

Expected: cero resultados.

- [ ] **Step 5: Commit**

Run:
```bash
cd C:/proyectos/python/ossflow-platform
git add -A
git commit -m "refactor(docker): instalar ossflow-service-kit desde Git, usar ossflow-base"
```

#### Task 10: Reescribir `docker-compose.yml`

**Files:**
- Modify: `C:/proyectos/python/ossflow-platform/docker-compose.yml`
- Modify: `C:/proyectos/python/ossflow-platform/docker-compose.override.yml`

> **Cambios necesarios:**
> 1. Renombrar todos los nombres de servicio: `processor-api → ossflow-api`, `chapter-splitter → ossflow-splitter`, `subtitle-generator → ossflow-subtitle`, `dubbing-generator → ossflow-dubbing`, `telegram-fetcher → ossflow-telegram`, `processor-frontend` ← **ELIMINAR del compose** (frontend va a otro repo)
> 2. Renombrar los `dockerfile:` paths: `./processor-api/Dockerfile` → `./ossflow-api/Dockerfile`, etc.
> 3. Eliminar el bloque `bjj-base` entero (la imagen base la construye `ossflow-core`)
> 4. Renombrar volumen `bjj-db` → `ossflow-db`
> 5. Renombrar network `bjj_net` → `ossflow_net`
> 6. Renombrar var env `BJJ_DB_PATH` → `OSSFLOW_DB_PATH`
> 7. Eliminar el servicio `chapter-splitter` (lean) — se va a `ossflow-scrapper`. Mantener solo `chapter-splitter-signal` renombrado a `ossflow-splitter` (sin profile, ahora es default)
> 8. Actualizar `SPLITTER_URL` y eliminar `SIGNAL_SPLITTER_URL` (ahora hay un único splitter), y añadir `SCRAPPER_URL=http://ossflow-scrapper:8001` apuntando al contenedor del nuevo repo (que se levanta con su propio compose)

- [ ] **Step 1: Reescribir `docker-compose.yml`**

Replace `C:/proyectos/python/ossflow-platform/docker-compose.yml` con el siguiente contenido:

```yaml
# ============================================================
# OSSFlow Platform — Docker Compose
# ============================================================
# Levanta todo el backend: docker compose up -d
#
# Servicios:
#   - ossflow-api:        FastAPI gateway          → :8000
#   - ossflow-splitter:   Detector por señales (GPU) → :8001
#   - ossflow-subtitle:   WhisperX (GPU)           → :8002
#   - ossflow-dubbing:    ElevenLabs cloud TTS     → :8003
#   - ossflow-telegram:   Descarga de media        → :8004
#   - ollama:             LLM local (opt-in)       → :11434
#
# Requiere:
#   - ossflow-base:latest construido desde el repo ossflow-core.
#   - ossflow-scrapper levantado en su propio compose si se usa el modo
#     scrapper (default). Conecta vía la network compartida ossflow_net.
# ============================================================

services:

  ossflow-api:
    build:
      context: .
      dockerfile: ./ossflow-api/Dockerfile
    container_name: ossflow-api
    ports:
      - "8000:8000"
    volumes:
      - config-data:/data/config
      - ossflow-db:/data/db
      - voice-profiles:/app/voice_profiles/samples
      - search-index:/app/search
      - nas-library:/media:rw
    environment:
      - PYTHONUNBUFFERED=1
      - CONFIG_DIR=/data/config
      - OSSFLOW_DB_PATH=/data/db/ossflow.db
      - MEDIA_ROOT=/media
      - SPLITTER_URL=http://ossflow-splitter:8001
      - SCRAPPER_URL=http://ossflow-scrapper:8001
      - SUBS_URL=http://ossflow-subtitle:8002
      - DUBBING_URL=http://ossflow-dubbing:8003
      - TELEGRAM_FETCHER_URL=http://ossflow-telegram:8004
      - OLLAMA_BASE_URL=${OLLAMA_BASE_URL:-http://ollama:11434}
      - OLLAMA_URL=${OLLAMA_BASE_URL:-http://ollama:11434}
      - ELEVENLABS_API_KEY=${ELEVENLABS_API_KEY:-}
    depends_on:
      ossflow-splitter:
        condition: service_healthy
      ossflow-subtitle:
        condition: service_healthy
      ossflow-dubbing:
        condition: service_healthy
    networks:
      - ossflow_net
    privileged: true
    restart: unless-stopped

  ossflow-splitter:
    build:
      context: .
      dockerfile: ./ossflow-splitter/Dockerfile
    container_name: ossflow-splitter
    ports:
      - "8001:8001"
    volumes:
      - nas-library:/media:rw
      - hf-cache:/models/huggingface
      - torch-cache:/models/torch
      - model-cache:/models/cache
    environment:
      - PYTHONUNBUFFERED=1
      - NVIDIA_VISIBLE_DEVICES=all
    networks:
      - ossflow_net
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://localhost:8001/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 30s
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  ossflow-subtitle:
    build:
      context: .
      dockerfile: ./ossflow-subtitle/Dockerfile
    container_name: ossflow-subtitle
    ports:
      - "8002:8002"
    volumes:
      - nas-library:/media:rw
      - hf-cache:/models/huggingface
      - torch-cache:/models/torch
      - model-cache:/models/cache
    environment:
      - PYTHONUNBUFFERED=1
      - NVIDIA_VISIBLE_DEVICES=all
      - OLLAMA_BASE_URL=${OLLAMA_BASE_URL:-http://ollama:11434}
    networks:
      - ossflow_net
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://localhost:8002/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 60s
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  ossflow-dubbing:
    build:
      context: .
      dockerfile: ./ossflow-dubbing/Dockerfile
    container_name: ossflow-dubbing
    ports:
      - "8003:8003"
    volumes:
      - nas-library:/media:rw
      - hf-cache:/models/huggingface
      - torch-cache:/models/torch
      - model-cache:/models/cache
      - ./ossflow-dubbing/voices:/voices:rw
      - ${S2PRO_MODEL_DIR:-./models/s2pro-gguf}:/models/s2pro:ro
    environment:
      - PYTHONUNBUFFERED=1
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - TORCH_HOME=/models/torch
      - HF_HOME=/models/huggingface
      - XDG_CACHE_HOME=/models/cache
      - DUBBING_MODEL_VOICE_PATH=${DUBBING_MODEL_VOICE_PATH:-}
      - ELEVENLABS_API_KEY=${ELEVENLABS_API_KEY:-}
      - DUBBING_TTS_ENGINE=${DUBBING_TTS_ENGINE:-}
      - S2PRO_GGUF_PATH=${S2PRO_GGUF_PATH:-/models/s2pro/s2-pro-q6_k.gguf}
      - S2PRO_TOKENIZER_PATH=${S2PRO_TOKENIZER_PATH:-/models/s2pro/tokenizer.json}
    networks:
      - ossflow_net
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://localhost:8003/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 60s
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu, compute, utility]
    restart: unless-stopped

  ossflow-telegram:
    build:
      context: .
      dockerfile: ./ossflow-telegram/Dockerfile
    container_name: ossflow-telegram
    ports:
      - "8004:8004"
    volumes:
      - tg-session:/data/session
      - tg-cache:/data/cache
      - ossflow-db:/data/db
      - nas-library:/media:rw
    environment:
      - PYTHONUNBUFFERED=1
      - PROCESSOR_API_URL=http://ossflow-api:8000
      - OSSFLOW_DB_PATH=/data/db/ossflow.db
    networks:
      - ossflow_net
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://localhost:8004/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 15s
    restart: unless-stopped

  ollama:
    profiles: ["ollama"]
    image: ollama/ollama:latest
    container_name: ossflow-ollama
    restart: unless-stopped
    networks:
      - ossflow_net
    ports:
      - "11434:11434"
    volumes:
      - ollama-models:/root/.ollama
      - ./ollama/entrypoint.sh:/entrypoint.sh:ro
    environment:
      - OLLAMA_KEEP_ALIVE=2m
      - OLLAMA_HOST=0.0.0.0:11434
    entrypoint: ["/bin/bash", "-c", "tr -d '\\r' < /entrypoint.sh | bash"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    healthcheck:
      test: ["CMD-SHELL", "curl -fsS http://localhost:11434/api/tags | grep -q qwen2.5:7b-instruct-q4_K_M"]
      interval: 30s
      timeout: 10s
      retries: 30
      start_period: 1200s

volumes:
  config-data:
  ossflow-db:
  voice-profiles:
  search-index:
  hf-cache:
  torch-cache:
  model-cache:
  tg-session:
  tg-cache:
  ollama-models:
  nas-library:
    driver: local
    driver_opts:
      type: ${NAS_VOLUME_TYPE:-cifs}
      device: ${NAS_VOLUME_DEVICE:-//${NAS_HOST}/${NAS_SHARE}}
      o: ${NAS_VOLUME_OPTS:-username=${NAS_USER},password=${NAS_PASS},uid=0,gid=0,vers=3.0,cache=strict,iocharset=utf8}

networks:
  ossflow_net:
    name: ossflow_net
    driver: bridge
```

> **Nota sobre la network:** se añade `name: ossflow_net` explícito porque el repo `ossflow-scrapper` declarará la misma network como `external: true` para conectarse.

- [ ] **Step 2: Reescribir `docker-compose.override.yml`**

Read el archivo actual y reescribirlo aplicando los mismos renombrados (servicios `processor-api` → `ossflow-api`, etc.). Si el override actual solo tenía ajustes menores, mantener la lógica idéntica con los nuevos nombres. Si tras la revisión el override ya no aporta nada útil, vaciarlo dejando solo:

```yaml
services: {}
```

- [ ] **Step 3: Verificar que no quedan referencias a los nombres viejos en compose**

Run:
```bash
cd C:/proyectos/python/ossflow-platform
grep -n "bjj\|processor-api\|processor-frontend\|chapter-splitter\|subtitle-generator\|dubbing-generator\|telegram-fetcher" docker-compose.yml docker-compose.override.yml
```

Expected: cero resultados (excepto comentarios históricos que no afecten al parsing — pero idealmente cero).

- [ ] **Step 4: Validar la sintaxis del compose**

Run:
```bash
cd C:/proyectos/python/ossflow-platform
docker compose config --quiet
```

Expected: sin output (silencioso) y exit code 0. Si hay errores, leerlos y corregir.

- [ ] **Step 5: Commit**

Run:
```bash
cd C:/proyectos/python/ossflow-platform
git add docker-compose.yml docker-compose.override.yml
git commit -m "refactor(compose): renombrar servicios, network y vars a ossflow_*"
```

#### Task 11: Migrar la variable de entorno `BJJ_DB_PATH` → `OSSFLOW_DB_PATH` en código

**Files:**
- Modify: cualquier `*.py` que lea `BJJ_DB_PATH` (probablemente en `ossflow_service_kit/db/engine.py` y código de telegram-fetcher)

> **Estrategia de compatibilidad:** durante esta tarea cambiamos las lecturas a `OSSFLOW_DB_PATH` con fallback a `BJJ_DB_PATH` para los entornos que aún no se hayan actualizado. **NO** mantenemos compatibilidad indefinida — solo durante el split. Una tarea posterior eliminará el fallback.

- [ ] **Step 1: Buscar todas las lecturas de `BJJ_DB_PATH`**

Run:
```bash
cd C:/proyectos/python/ossflow-platform
grep -rn "BJJ_DB_PATH" --include="*.py" --include="*.yml" --include="*.yaml" --include="Dockerfile*" --include="*.md"
```

Anotar cada archivo y línea.

- [ ] **Step 2: En cada archivo `.py` con `os.environ.get("BJJ_DB_PATH"...)` (o variantes), añadir fallback**

Patrón viejo:
```python
db_path = os.environ.get("BJJ_DB_PATH", "/data/db/bjj.db")
```

Patrón nuevo:
```python
# OSSFLOW_DB_PATH es el nombre canónico; BJJ_DB_PATH se mantiene como fallback
# durante la migración del split de repos. Se eliminará en una versión futura.
db_path = os.environ.get("OSSFLOW_DB_PATH") or os.environ.get("BJJ_DB_PATH", "/data/db/ossflow.db")
```

> Nota: el default cambia de `/data/db/bjj.db` a `/data/db/ossflow.db`. Sin embargo, **el archivo físico en el volumen sigue llamándose `bjj.db` hasta que se ejecute la Task 14 (smoke test de pipeline)**. Por eso en compose se sigue apuntando a `OSSFLOW_DB_PATH=/data/db/ossflow.db` Y se hace una copia del archivo en el step de validación. Esto se gestiona en el Task 14.

> **Decisión simplificada para evitar este conflicto:** mantener el filename físico como `bjj.db` durante esta migración. Cambiar el default Python a `/data/db/bjj.db` (igual que antes), y solo renombrar la **variable de entorno**. El archivo de BD físico se renombra en una migración posterior a este plan.

Aplicar entonces este patrón:

```python
db_path = os.environ.get("OSSFLOW_DB_PATH") or os.environ.get("BJJ_DB_PATH", "/data/db/bjj.db")
```

Y revisar el `docker-compose.yml` recién escrito en Task 10: cambiar `OSSFLOW_DB_PATH=/data/db/ossflow.db` a `OSSFLOW_DB_PATH=/data/db/bjj.db` para mantener el archivo físico.

- [ ] **Step 3: Aplicar el cambio al `docker-compose.yml`**

Edit `C:/proyectos/python/ossflow-platform/docker-compose.yml`:

Cambiar las dos ocurrencias de:
```
- OSSFLOW_DB_PATH=/data/db/ossflow.db
```

por:
```
- OSSFLOW_DB_PATH=/data/db/bjj.db
```

- [ ] **Step 4: Verificar**

Run:
```bash
cd C:/proyectos/python/ossflow-platform
grep -rn "BJJ_DB_PATH\|OSSFLOW_DB_PATH" --include="*.py" --include="*.yml" --include="*.yaml"
```

Expected: cada archivo Python muestra el patrón con fallback; los YAMLs solo muestran `OSSFLOW_DB_PATH=/data/db/bjj.db`.

- [ ] **Step 5: Commit**

Run:
```bash
cd C:/proyectos/python/ossflow-platform
git add -A
git commit -m "refactor(env): introducir OSSFLOW_DB_PATH con fallback a BJJ_DB_PATH"
```

#### Task 12: Verificar tests unitarios de cada servicio (sin Docker)

**Files:**
- Run-only

> Cada servicio tiene su propio `tests/`. Hasta que el kit esté disponible vía Git, los tests locales necesitan que `ossflow-service-kit` esté instalado en el venv. Para esta tarea instalamos el kit desde el filesystem local (no desde Git) y validamos que cada suite de tests pasa.

- [ ] **Step 1: Crear venv compartido para tests**

Run:
```bash
python -m venv C:/proyectos/python/ossflow-platform/.venv
C:/proyectos/python/ossflow-platform/.venv/Scripts/python.exe -m pip install --upgrade pip
```

- [ ] **Step 2: Instalar `ossflow-service-kit` desde filesystem local**

Run:
```bash
C:/proyectos/python/ossflow-platform/.venv/Scripts/python.exe -m pip install -e "C:/proyectos/python/ossflow-core/ossflow_service_kit[test]"
```

- [ ] **Step 3: Instalar requirements de cada servicio y correr sus tests**

Para cada servicio (`ossflow-api`, `ossflow-splitter`, `ossflow-subtitle`, `ossflow-dubbing`, `ossflow-telegram`):

```bash
SVC=ossflow-api  # repetir con cada uno
cd C:/proyectos/python/ossflow-platform/$SVC
C:/proyectos/python/ossflow-platform/.venv/Scripts/python.exe -m pip install -r requirements.txt
C:/proyectos/python/ossflow-platform/.venv/Scripts/python.exe -m pytest tests/ -v --ignore=tests/test_backend_client.py
```

> Nota: `test_backend_client.py` se ignora porque ese test hace HTTP real contra los backends y no funciona offline.

Expected: todos los tests pasan. **Si algún test falla por imports `bjj_*`**, parar y resolver — significa que el find/replace del Task 8 no fue exhaustivo.

- [ ] **Step 4: Documentar cualquier test que requiera fixture especial o haya quedado roto**

Si un test no se puede pasar localmente (ej. requiere GPU, o credenciales), anotar en un commit message del próximo commit. **No skipear tests sin explicación.** Si un test falla por una razón no relacionada con el rename, abrir un issue (en el repo origen archivado o en el nuevo repo) y referenciarlo.

- [ ] **Step 5: Commit (si hubo cambios)**

Run:
```bash
cd C:/proyectos/python/ossflow-platform
git add -A
git diff --cached --quiet || git commit -m "fix: ajustes mínimos derivados del split (ver detalles)"
```

> Si no hay cambios staged, no se crea commit (la lógica `git diff --cached --quiet` evita commit vacío).

#### Task 13: Build completo de imágenes Docker

**Files:**
- Build-only

- [ ] **Step 1: Build de cada servicio**

Run:
```bash
cd C:/proyectos/python/ossflow-platform
docker compose build
```

Expected: build exitoso de los 5 servicios. Si algún Dockerfile falla con `pip install ossflow-service-kit @ git+...`, las causas típicas son:
- Repo `ossflow-core` privado y falta auth → resolver con deploy keys o token de Personal Access Token en una variable `GITHUB_TOKEN` y usar `git+https://${GITHUB_TOKEN}@github.com/...`
- Tag `v0.1.0` no fue pusheado → resolver con `git push origin v0.1.0` desde `ossflow-core`

- [ ] **Step 2: Verificar imágenes generadas**

Run:
```bash
docker images | grep ossflow
```

Expected: imágenes `ossflow-platform-ossflow-api`, `ossflow-platform-ossflow-splitter`, etc.

- [ ] **Step 3: Commit (si hubo cambios)**

Si hubo que ajustar Dockerfiles para que el build funcione:

```bash
cd C:/proyectos/python/ossflow-platform
git add -A
git diff --cached --quiet || git commit -m "fix(docker): ajustes para que el build remoto del kit funcione"
```

#### Task 14: Smoke test end-to-end con un instruccional real

**Files:**
- Validation-only

> Este test valida que el split no rompió nada funcional. Se levanta el platform, se carga un instruccional pequeño, y se ejecuta un pipeline completo. Aún sin `ossflow-scrapper` separado (que se hace en la Fase 3), el splitter signal del platform sigue funcionando.

- [ ] **Step 1: Restaurar BD desde backup**

Run:
```bash
cd C:/proyectos/python/ossflow-platform
docker volume create ossflow-platform_ossflow-db
docker run --rm -v ossflow-platform_ossflow-db:/data/db -v C:/proyectos/python/.backups:/backup alpine cp /backup/bjj-pre-refactor.db /data/db/bjj.db
```

Expected: la BD del backup queda dentro del nuevo volumen `ossflow-platform_ossflow-db`.

- [ ] **Step 2: Levantar el stack**

Run:
```bash
cd C:/proyectos/python/ossflow-platform
docker compose up -d
```

Expected: todos los contenedores en estado `healthy` tras 1-2 minutos. Verificar con `docker compose ps`.

- [ ] **Step 3: Probar `/api/settings` y `/api/library`**

Run:
```bash
curl -fsS http://localhost:8000/api/settings | head -c 200
curl -fsS http://localhost:8000/api/library | head -c 200
```

Expected: respuestas JSON válidas (no errores 500).

- [ ] **Step 4: Probar `/health` y `/gpu` de cada backend GPU**

Run:
```bash
curl -fsS http://localhost:8001/health
curl -fsS http://localhost:8002/health
curl -fsS http://localhost:8003/health
curl -fsS http://localhost:8001/gpu
curl -fsS http://localhost:8002/gpu
curl -fsS http://localhost:8003/gpu
```

Expected: respuestas 200 con info esperada.

- [ ] **Step 5: Verificar logs sin errores**

Run:
```bash
docker compose logs --tail=50 ossflow-api ossflow-splitter ossflow-subtitle ossflow-dubbing ossflow-telegram | grep -i "error\|traceback" | head -20
```

Expected: cero o muy pocos errores no relacionados con el split. Cualquier error con `bjj_service_kit` en la traza es un bug del rename.

- [ ] **Step 6: Bajar el stack**

Run:
```bash
cd C:/proyectos/python/ossflow-platform
docker compose down
```

#### Task 15: Push de `ossflow-platform`

**Files:**
- Push remoto

- [ ] **Step 1: Push de la rama main**

Run:
```bash
cd C:/proyectos/python/ossflow-platform
git branch -M main
git push -u origin main
```

Expected: rama `main` pusheada.

---

### Fase 3: `ossflow-scrapper` (nuevo repo, no el monorepo origen)

#### Task 16: Crear estructura local de `ossflow-scrapper`

**Files:**
- Create: `C:/proyectos/python/ossflow-scrapper-new/`

- [ ] **Step 1: Crear directorio del repo**

Run:
```bash
mkdir -p C:/proyectos/python/ossflow-scrapper-new
cd C:/proyectos/python/ossflow-scrapper-new
git init
git remote add origin https://github.com/<GITHUB_USER>/ossflow-scrapper.git
```

- [ ] **Step 2: Copiar el código del modo oracle del splitter**

Run (PowerShell):
```powershell
$src = "C:/proyectos/python/ossflow-scrapper/chapter-splitter"
$dst = "C:/proyectos/python/ossflow-scrapper-new"
Copy-Item "$src/app_oracle.py" "$dst/app.py"
Copy-Item "$src/Dockerfile.oracle" "$dst/Dockerfile"
Copy-Item "$src/requirements.txt" "$dst/requirements.txt" -ErrorAction SilentlyContinue
```

> **Nota:** `app_oracle.py` se renombra a `app.py` para que el Dockerfile use el nombre canónico. Si el módulo importa funciones desde otros archivos de `chapter-splitter/` (proveedores, parsers), copiarlos también. Verificar imports antes de continuar.

- [ ] **Step 3: Verificar y copiar dependencias**

Run:
```bash
cd C:/proyectos/python/ossflow-scrapper-new
grep -n "^from\|^import" app.py | head -30
```

Identificar cualquier `from <local_module>` que apunte a archivos que no se copiaron. Si los hay, copiarlos al nuevo repo manteniendo la misma estructura relativa.

- [ ] **Step 4: Crear `.gitignore` y `README.md`**

Create `C:/proyectos/python/ossflow-scrapper-new/.gitignore`:

```
__pycache__/
*.pyc
.pytest_cache/
.venv/
.env
*.egg-info/
```

Create `C:/proyectos/python/ossflow-scrapper-new/README.md`:

```markdown
# ossflow-scrapper

Servicio de scraping de proveedores de instruccionales para el ecosistema OSSFlow.

## Proveedores soportados

- BJJ Fanatics (actual)
- Grappling Industries (fase 2)
- FloGrappling (fase 2)
- YouTube (fase 2)

## Endpoints

- `POST /scrape` — extrae capítulos y timestamps de una URL
- `GET /search?q=...` — búsqueda cross-provider
- `GET /health`, `/gpu`, `/logs`

## Arranque

```bash
docker compose up -d
```

Se conecta a la network compartida `ossflow_net` definida por `ossflow-platform`.
```

- [ ] **Step 5: Renombrar imports `bjj_service_kit` → `ossflow_service_kit`**

Run (PowerShell):
```powershell
$root = "C:/proyectos/python/ossflow-scrapper-new"
$files = Get-ChildItem -Path $root -Recurse -Include *.py
foreach ($f in $files) {
  (Get-Content $f.FullName) -replace 'bjj_service_kit', 'ossflow_service_kit' | Set-Content $f.FullName
}
```

- [ ] **Step 6: Actualizar Dockerfile**

Read `C:/proyectos/python/ossflow-scrapper-new/Dockerfile` y aplicar:
- `FROM bjj-base:latest` → `FROM ossflow-base:latest` (si lo usa; si es lean image, mantener su `FROM` original)
- Reemplazar el bloque `COPY bjj_service_kit ... && pip install ...` por:

```dockerfile
RUN pip install --no-cache-dir "ossflow-service-kit @ git+https://github.com/<GITHUB_USER>/ossflow-core@v0.1.0#subdirectory=ossflow_service_kit"
```

- Si hubiera `COPY app_oracle.py app.py` o similar, ajustar a `COPY app.py /app/app.py`.

- [ ] **Step 7: Crear `docker-compose.yml` mínimo**

Create `C:/proyectos/python/ossflow-scrapper-new/docker-compose.yml`:

```yaml
services:
  ossflow-scrapper:
    build:
      context: .
      dockerfile: ./Dockerfile
    container_name: ossflow-scrapper
    ports:
      - "8005:8001"   # 8005 en host para no chocar con ossflow-splitter
    environment:
      - PYTHONUNBUFFERED=1
    networks:
      - ossflow_net
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://localhost:8001/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 15s
    restart: unless-stopped

networks:
  ossflow_net:
    external: true
```

> **Nota:** la network `ossflow_net` la crea `ossflow-platform`. Por eso aquí va `external: true`. El orden de arranque es: levantar primero el platform, luego el scrapper.

- [ ] **Step 8: Verificar grep**

Run:
```bash
cd C:/proyectos/python/ossflow-scrapper-new
grep -rn "bjj_service_kit\|bjj-base\|bjj_base" .
```

Expected: cero resultados.

- [ ] **Step 9: Commit inicial**

Run:
```bash
cd C:/proyectos/python/ossflow-scrapper-new
git add .
git commit -m "chore: importar código del modo oracle desde monorepo"
```

#### Task 17: Build y smoke test de `ossflow-scrapper`

**Files:**
- Build/run only

- [ ] **Step 1: Levantar primero `ossflow-platform` (para que la network exista)**

Run:
```bash
cd C:/proyectos/python/ossflow-platform
docker compose up -d
```

Expected: stack platform arriba, network `ossflow_net` creada.

- [ ] **Step 2: Build del scrapper**

Run:
```bash
cd C:/proyectos/python/ossflow-scrapper-new
docker compose build
```

Expected: build exitoso.

- [ ] **Step 3: Levantar scrapper**

Run:
```bash
cd C:/proyectos/python/ossflow-scrapper-new
docker compose up -d
```

Expected: contenedor `ossflow-scrapper` healthy tras ~30s.

- [ ] **Step 4: Smoke test**

Run:
```bash
curl -fsS http://localhost:8005/health
```

Expected: respuesta 200.

- [ ] **Step 5: Validar conectividad desde ossflow-api**

Run:
```bash
docker exec ossflow-api curl -fsS http://ossflow-scrapper:8001/health
```

Expected: respuesta 200 (la API resuelve el scrapper por nombre de contenedor en la network compartida).

- [ ] **Step 6: Bajar ambos stacks**

Run:
```bash
cd C:/proyectos/python/ossflow-scrapper-new
docker compose down
cd C:/proyectos/python/ossflow-platform
docker compose down
```

#### Task 18: Push de `ossflow-scrapper`

**Files:**
- Push remoto

- [ ] **Step 1: Push**

Run:
```bash
cd C:/proyectos/python/ossflow-scrapper-new
git branch -M main
git push -u origin main
```

---

### Fase 4: `ossflow-studio`

#### Task 19: Crear estructura local de `ossflow-studio`

**Files:**
- Create: `C:/proyectos/python/ossflow-studio/`

- [ ] **Step 1: Crear directorio del repo**

Run:
```bash
mkdir -p C:/proyectos/python/ossflow-studio
cd C:/proyectos/python/ossflow-studio
git init
git remote add origin https://github.com/<GITHUB_USER>/ossflow-studio.git
```

- [ ] **Step 2: Copiar `processor-frontend/` → `ossflow-frontend/`**

Run (PowerShell):
```powershell
$src = "C:/proyectos/python/ossflow-scrapper/processor-frontend"
$dst = "C:/proyectos/python/ossflow-studio/ossflow-frontend"
Copy-Item -Recurse $src $dst
Remove-Item -Recurse -Force "$dst/node_modules" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "$dst/dist" -ErrorAction SilentlyContinue
```

- [ ] **Step 3: Crear `.gitignore` y `README.md` en la raíz**

Create `C:/proyectos/python/ossflow-studio/.gitignore`:

```
node_modules/
dist/
.env
.env.local
.idea/
.vscode/
```

Create `C:/proyectos/python/ossflow-studio/README.md`:

```markdown
# ossflow-studio

SPA React de gestión del ecosistema OSSFlow.

## Tech Stack

- React 18 + Vite
- Tailwind CSS
- Zustand (state)
- React Router

## Desarrollo

```bash
cd ossflow-frontend
npm install
npm run dev
```

Por defecto se conecta a `http://localhost:8000` (ossflow-api). Para apuntar a otro backend, ajustar `vite.config.js`.

## Producción

```bash
npm run build
docker build -t ossflow-frontend .
```
```

- [ ] **Step 4: Verificar package.json del frontend**

Read `C:/proyectos/python/ossflow-studio/ossflow-frontend/package.json` y, si el campo `name` es `processor-frontend`, cambiarlo a `ossflow-frontend`.

- [ ] **Step 5: Buscar referencias a nombres viejos en código frontend**

Run:
```bash
cd C:/proyectos/python/ossflow-studio
grep -rn "processor-api\|processor-frontend\|bjj" --include="*.js" --include="*.jsx" --include="*.json" --include="*.html" --include="*.config.*" | head -30
```

Si hay referencias en URLs hardcodeadas, comentarios o nombres de paquete, **listarlas** y aplicar criterio:
- URLs como `http://localhost:8000` no necesitan cambio (es la URL del API, no el nombre)
- Comentarios mencionando "BJJ Instructional Processor" pueden mantenerse (es el dominio del producto, no el nombre técnico) o cambiarse a "OSSFlow"
- Imports relativos no afectados

> **Decisión por defecto:** mantener los textos de UI ("BJJ Instructional Processor") como están. Solo se cambian referencias técnicas a nombres de servicio.

- [ ] **Step 6: Smoke test del build**

Run:
```bash
cd C:/proyectos/python/ossflow-studio/ossflow-frontend
npm install
npm run build
```

Expected: build sin errores. Genera `dist/`.

- [ ] **Step 7: Commit inicial**

Run:
```bash
cd C:/proyectos/python/ossflow-studio
echo "dist/" >> ossflow-frontend/.gitignore  # asegurar que dist no se commitea
git add .
git commit -m "chore: importar frontend desde monorepo, renombrar a ossflow-frontend"
```

#### Task 20: Push de `ossflow-studio`

**Files:**
- Push remoto

- [ ] **Step 1: Push**

Run:
```bash
cd C:/proyectos/python/ossflow-studio
git branch -M main
git push -u origin main
```

---

### Fase 5: Validación Final y Archivado

#### Task 21: Validación end-to-end con los 4 repos

**Files:**
- Validation-only

- [ ] **Step 1: Levantar el platform**

Run:
```bash
cd C:/proyectos/python/ossflow-platform
docker compose up -d
```

- [ ] **Step 2: Levantar el scrapper**

Run:
```bash
cd C:/proyectos/python/ossflow-scrapper-new
docker compose up -d
```

- [ ] **Step 3: Servir el frontend en dev**

Run (en otra terminal):
```bash
cd C:/proyectos/python/ossflow-studio/ossflow-frontend
npm run dev
```

Expected: dev server arriba en http://localhost:5173 (o el puerto que tenga configurado Vite).

- [ ] **Step 4: Test manual end-to-end**

Abrir http://localhost:5173 en navegador. Verificar:
- Settings cargan
- Library scan funciona
- Lista de instruccionales con pósters
- Detalle de un instruccional muestra capítulos
- Logs viewer muestra logs de los backends

> **No es necesario** ejecutar un pipeline completo aquí — eso se valida en la Fase 3 del refactor (Plan siguiente). Lo que importa en este plan es que **todo arranca y se comunica**.

- [ ] **Step 5: Bajar todo**

Run:
```bash
cd C:/proyectos/python/ossflow-scrapper-new && docker compose down
cd C:/proyectos/python/ossflow-platform && docker compose down
```

#### Task 22: Archivar el monorepo origen

**Files:**
- Modify: GitHub repo `ossflow-scrapper` (origen) — renombrar y archivar
- Modify: filesystem local — renombrar `ossflow-scrapper` → `ossflow-monorepo-archive`, y `ossflow-scrapper-new` → `ossflow-scrapper`

- [ ] **Step 1: Renombrar el repo en GitHub**

Run:
```bash
gh repo rename ossflow-monorepo-archive --repo <GITHUB_USER>/ossflow-scrapper
```

Expected: el repo origen ahora se llama `ossflow-monorepo-archive` en GitHub. Esto libera el nombre `ossflow-scrapper` para el repo nuevo.

- [ ] **Step 2: Push del repo nuevo de scrapper (que ya tiene el remoto correcto)**

Run:
```bash
cd C:/proyectos/python/ossflow-scrapper-new
git remote -v
```

Expected: el remoto apunta a `https://github.com/<GITHUB_USER>/ossflow-scrapper.git`. Si es así y ya está pusheado (Task 18), no hay nada que hacer aquí.

Si el push de Task 18 falló porque el repo origen aún tenía ese nombre, repetir ahora:

```bash
cd C:/proyectos/python/ossflow-scrapper-new
git push -u origin main
```

- [ ] **Step 3: Renombrar directorios locales**

Run (PowerShell):
```powershell
Rename-Item C:/proyectos/python/ossflow-scrapper C:/proyectos/python/ossflow-monorepo-archive
Rename-Item C:/proyectos/python/ossflow-scrapper-new C:/proyectos/python/ossflow-scrapper
```

- [ ] **Step 4: Archivar el repo en GitHub**

Run:
```bash
gh repo archive <GITHUB_USER>/ossflow-monorepo-archive --yes
```

Expected: el repo aparece como archivado en GitHub. Sigue siendo accesible en read-only.

- [ ] **Step 5: Documentar el split en el README del archive**

Edit el README del repo archivado (vía web o `gh repo clone`) para añadir al inicio:

```markdown
# OSSFlow Monorepo (archivado)

Este repositorio fue dividido el 2026-04-29 en cuatro repos independientes:

- [`ossflow-core`](https://github.com/<GITHUB_USER>/ossflow-core) — infraestructura compartida
- [`ossflow-platform`](https://github.com/<GITHUB_USER>/ossflow-platform) — backend orquestado
- [`ossflow-scrapper`](https://github.com/<GITHUB_USER>/ossflow-scrapper) — scraping de proveedores
- [`ossflow-studio`](https://github.com/<GITHUB_USER>/ossflow-studio) — frontend SPA

Ver el spec del split en `docs/superpowers/specs/2026-04-29-ossflow-architecture-refactor-design.md`.
```

#### Task 23: Limpieza final

**Files:**
- Cleanup

- [ ] **Step 1: Eliminar imágenes Docker viejas**

Run:
```bash
docker images | grep -E "bjj-|processor-|chapter-splitter|subtitle-generator|dubbing-generator|telegram-fetcher" | awk '{print $3}' | xargs -r docker rmi -f
```

Expected: imágenes obsoletas eliminadas. Si alguna está en uso por un contenedor parado, primero `docker rm <container>` y reintentar.

- [ ] **Step 2: Eliminar volúmenes huérfanos del compose viejo**

Run:
```bash
docker volume ls | grep "ossflow-scrapper_"
```

Si aparecen volúmenes con prefijo `ossflow-scrapper_` (del antiguo nombre del monorepo), confirmar uno por uno antes de borrar — pueden tener datos de la BD migrada.

> **No borrar automáticamente.** El backup ya existe en `C:/proyectos/python/.backups/bjj-pre-refactor.db`, pero los volúmenes pueden tener config-data, voice-profiles, etc. que aún sean útiles. Borrar solo tras confirmar con el usuario.

- [ ] **Step 3: Crear documento de cierre**

Create `C:/proyectos/python/ossflow-platform/docs/MIGRATION-NOTES.md`:

```markdown
# Notas del Split de Repos (2026-04-29)

## Cambios principales

- Monorepo `ossflow-scrapper` (originalmente `bjj-processor-v2`) dividido en 4 repos.
- Identificadores `bjj_*` renombrados a `ossflow_*` en todo el código.
- Imagen Docker base `bjj-base` → `ossflow-base`.
- Paquete Python `bjj_service_kit` → `ossflow_service_kit`, distribuido vía Git tag.
- Network compose `bjj_net` → `ossflow_net`.
- Variable env `BJJ_DB_PATH` → `OSSFLOW_DB_PATH` (con fallback a `BJJ_DB_PATH` durante una transición).
- Servicios renombrados:
  - `processor-api` → `ossflow-api`
  - `chapter-splitter` (signal) → `ossflow-splitter`
  - `chapter-splitter` (oracle / lean) → repo independiente `ossflow-scrapper`
  - `subtitle-generator` → `ossflow-subtitle`
  - `dubbing-generator` → `ossflow-dubbing`
  - `telegram-fetcher` → `ossflow-telegram`
  - `processor-frontend` → repo independiente `ossflow-studio` / `ossflow-frontend`
- Eliminado el módulo `burn_subs` (funcionalidad absorbida por `dubbing`).

## Pendiente (próximos planes)

- Refactor interno de `ossflow-api` con Vertical Slice (ver spec sección 3.3).
- Refactor interno de los demás servicios según su patrón asignado (ver spec sección 3.2).
- Eliminar el fallback `BJJ_DB_PATH` cuando ya no haya entornos legacy.
- Renombrar el archivo físico de la BD `bjj.db` → `ossflow.db` mediante una migración controlada.
```

- [ ] **Step 4: Commit del documento de notas en `ossflow-platform`**

Run:
```bash
cd C:/proyectos/python/ossflow-platform
git add docs/MIGRATION-NOTES.md
git commit -m "docs: añadir notas de migración del split de repos"
git push
```

---

## Resumen ejecutable

Total de tareas: **23 tareas, ~110 steps**.

Distribución por fase:
- Fase 0 (Preparación): 2 tareas
- Fase 1 (`ossflow-core`): 4 tareas
- Fase 2 (`ossflow-platform`): 9 tareas
- Fase 3 (`ossflow-scrapper`): 3 tareas
- Fase 4 (`ossflow-studio`): 2 tareas
- Fase 5 (Validación + Archivado): 3 tareas

Hitos de validación:
- Tras Task 6: el kit se instala vía pip desde GitHub
- Tras Task 12: tests unitarios de los 5 servicios pasan en local
- Tras Task 14: pipeline e2e funciona con los nuevos nombres en `ossflow-platform`
- Tras Task 17: scrapper se conecta al platform via network compartida
- Tras Task 21: frontend abre y dialoga con el platform completo
- Tras Task 22: monorepo origen archivado, los 4 repos viven independientemente

## Estado siguiente

Una vez completado este plan, cada repo está en su sitio con código funcional pero **arquitectónicamente igual al original**. El siguiente plan (a redactar después) aplicará Vertical Slice en `ossflow-api` y los demás patrones definidos en el spec.
