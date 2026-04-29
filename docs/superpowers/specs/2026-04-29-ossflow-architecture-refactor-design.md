# Refactor Arquitectónico OSSFlow — Diseño

**Fecha:** 2026-04-29
**Estado:** Aprobado, pendiente de plan de implementación
**Autor:** Adrián + Claude (rol arquitecto)

---

## 1. Contexto y Motivación

### Estado actual

El proyecto vive en un único monorepo (`ossflow-scrapper`, históricamente `bjj-processor-v2`) con seis servicios Python + un frontend React. Tras meses de desarrollo iterativo, la base de código presenta los siguientes síntomas:

- **Archivos gigantes:** `processor-api/api/app.py` (~1790 líneas) y `processor-api/api/pipeline.py` (~1783 líneas) concentran lógica HTTP, reglas de negocio, acceso a disco e integración con microservicios sin separación clara.
- **Responsabilidades difusas:** `bjj_service_kit` mezcla infraestructura transversal con schemas de dominio y migraciones de base de datos.
- **Falta de patrones:** no existe un patrón arquitectónico explícito; cada feature se ha añadido siguiendo convenciones distintas.
- **Repositorio único:** servicios con ciclos de vida y consumidores diferentes comparten el mismo repositorio, dificultando el versionado independiente y la fase 2 (web "segundo cerebro" `ossflow`).
- **Onboarding difícil:** un nuevo desarrollador (o el propio autor tras un parón) no puede inferir las reglas del proyecto leyendo el código.

### Decisión sobre el lenguaje

Se evaluó migrar el backend a **Spring Boot** dado que el autor trabaja en Java profesionalmente. **Se descarta** porque:

1. Los modelos de IA del proyecto (WhisperX, Coqui/ElevenLabs, Demucs, S2-Pro, Ollama) tienen bindings nativos solo en Python. Migrar a Java exigiría mantener microservicios Python igualmente, añadiendo una capa Java innecesaria por encima.
2. El problema real **no es el lenguaje, es la arquitectura**. Python sigue siendo la elección correcta para este dominio.
3. Una migración total no es viable hoy desde el punto de vista de coste/beneficio.

### Objetivo del refactor

Reorganizar el código en repositorios coherentes y aplicar patrones arquitectónicos claros que permitan:

- Responsabilidades explícitas por servicio y por módulo
- Onboarding rápido (la estructura del repo enseña dónde va cada cosa)
- Habilitar la fase 2 (`ossflow` segundo cerebro) sin acoplamientos
- Mantener Python como lenguaje principal

---

## 2. Estructura de Repositorios

Se divide el monorepo actual en **cuatro repositorios independientes**:

| Repositorio | Contenido | Propósito |
|---|---|---|
| `ossflow-core` | `ossflow-base` (imagen Docker base CUDA) + `ossflow-service-kit` (paquete Python compartido) | Infraestructura común reutilizable y versionada |
| `ossflow-platform` | `ossflow-api` + `ossflow-splitter` + `ossflow-subtitle` + `ossflow-dubbing` + `ossflow-telegram` + `docker-compose` | Backend orquestado de procesamiento de instruccionales |
| `ossflow-scrapper` | Scraping BJJ Fanatics + futuros proveedores (FloGrappling, Grappling Industries, YouTube...) | Extracción de datos externos. Independiente para evolucionar al ritmo de los cambios de los proveedores y ser consumible desde la fase 2 |
| `ossflow-studio` | `ossflow-frontend` (React 18 + Tailwind + Zustand) | SPA actual de gestión |

**Justificación de la separación:**

- `ossflow-core` se separa porque cambia raramente y es dependencia transversal versionada.
- `ossflow-scrapper` se separa porque sus proveedores cambian con frecuencia (BJJ Fanatics actualiza su web), porque escalará en número de providers, y porque la fase 2 lo consumirá directamente sin pasar por el platform.
- `ossflow-studio` se separa porque su ciclo de despliegue (Vite build + Nginx) es completamente distinto a los backends Python (Docker CUDA).
- Los servicios del `platform` permanecen juntos porque se despliegan con el mismo `docker-compose`, comparten la base de datos SQLite (`/data/db/bjj.db`) y se versionan en bloque.

### Renombrado completo de paquetes

Todos los identificadores `bjj_*` y `bjj-*` se renombran a `ossflow_*` y `ossflow-*` respectivamente. El paquete Python compartido pasa a llamarse `ossflow_service_kit` (importable con guión bajo) aunque el repositorio sea `ossflow-core`.

---

## 3. Arquitectura Interna

### 3.1. `ossflow-service-kit` — Kit mínimo de infraestructura

El paquete compartido se reduce a **solo infraestructura transversal**. Todo lo que sea dominio o contratos de datos sale de aquí.

**Contenido aprobado:**

- `app_factory.py` — bootstrap FastAPI con CORS, lifespan, registro de routers comunes
- `log_bridge.py` + `events.py` — handler de logs con ring buffer y stream SSE
- Endpoints comunes: `/health`, `/gpu`, `/logs`
- `runner.py` — utilidad genérica para ejecución de jobs con seguimiento

**Lo que SALE de `ossflow-service-kit`:**

- ❌ Schemas Pydantic de dominio (van a cada servicio que los usa)
- ❌ Modelos SQLAlchemy y migraciones Alembic (van a `ossflow-api`, dueño de la BD)
- ❌ Cualquier lógica específica de pipelines, biblioteca, capítulos, etc.

**Razón:** un kit de infraestructura debe ser estable y agnóstico al dominio. Mezclar contratos de datos lo convierte en un acoplamiento difuso entre servicios.

### 3.2. Patrón arquitectónico por servicio

Se aplica **un patrón heterogéneo y pragmático** según la naturaleza de cada servicio:

| Servicio | Patrón | Justificación |
|---|---|---|
| `ossflow-api` | Vertical Slice | ~18 features independientes (pipeline, library, cleanup, duplicates, settings, telegram, metrics, etc.) |
| `ossflow-scrapper` | Vertical Slice | Múltiples providers presentes y futuros, cada uno con lógica propia |
| `ossflow-telegram` | Vertical Slice ligero | 3 responsabilidades genuinamente independientes (channels, downloads, auth) |
| `ossflow-subtitle` | Layered | Una responsabilidad descompuesta en pasos secuenciales (transcripción → postproceso → traducción) |
| `ossflow-dubbing` | Layered | Una responsabilidad descompuesta en pasos secuenciales (síntesis → alineamiento → muxing) |
| `ossflow-splitter` | Layered + Strategy | Una responsabilidad con dos estrategias de obtención de timestamps (signal ML, timestamps externos) que comparten el 80% del código |

**Principio rector:**

> **Vertical Slice** se justifica cuando hay 3+ features independientes con endpoints/responsabilidades propias y probabilidad alta de crecer en número.
>
> **Layered** es mejor cuando hay una responsabilidad descompuesta en pasos que siempre se ejecutan juntos.

Forzar Vertical Slice en servicios lineales genera carpetas vacías y boilerplate sin valor. Forzar Layered en servicios con muchas features dispersa la lógica.

### 3.3. Estructura Vertical Slice — `ossflow-api`

```
ossflow-api/
├── modules/
│   ├── pipeline/        # Ejecución de pipelines, steps, history, retry/cancel
│   ├── library/         # Scan, posters, capítulos, metadatos
│   ├── chapters/        # Renombrado y edición de capítulos
│   ├── settings/        # Configuración dinámica
│   ├── preflight/       # Comprobaciones previas al pipeline
│   ├── cleanup/         # SRTs huérfanos, _DOBLADO obsoletos
│   ├── duplicates/      # Detección de duplicados
│   ├── promote/         # Promoción de pistas a vídeo final
│   ├── subtitles/       # Endpoints SRT
│   ├── dubbing/         # Endpoints doblaje (Coqui/local)
│   ├── elevenlabs/      # Doblaje ElevenLabs (Studio + voces)
│   ├── scrapper/        # Consume ossflow-scrapper (antes "oracle")
│   ├── telegram/        # Canales, descargas, jobs telegram
│   ├── jobs/            # Background jobs (genérico)
│   ├── metrics/         # CPU/RAM/Disco/GPU
│   ├── logs/            # Ring buffer de logs
│   └── health/          # /health, /gpu (proxy a backends)
│
├── shared/
│   ├── paths.py         # to_container_path, traducción host↔contenedor
│   ├── events.py        # Normalizador de eventos SSE
│   └── exceptions.py    # Excepciones de dominio comunes
│
├── clients/             # Llamadas a otros microservicios
│   ├── splitter.py
│   ├── subtitle.py
│   ├── dubbing.py
│   ├── elevenlabs.py
│   ├── scrapper.py
│   └── ollama.py
│
├── infrastructure/      # Bootstrap puro (sin lógica de negocio)
│   ├── db.py            # Sesión SQLAlchemy
│   ├── config.py        # Variables de entorno, paths
│   └── lifespan.py      # Hooks startup/shutdown
│
├── main.py              # FastAPI app + registro de módulos
└── tests/
    ├── modules/         # Tests por módulo
    └── integration/     # Tests cross-módulo
```

**Cambios respecto al estado actual:**

- ❌ Eliminado `burn_subs/` — la lógica de quemado se ha absorbido en `dubbing`
- ✏️ `oracle/` renombrado a `scrapper/` para reflejar mejor su responsabilidad y alinearse con el repo `ossflow-scrapper`

### 3.4. Anatomía interna de un módulo

La estructura escala con la complejidad del módulo:

**Módulo simple** (`settings`, `metrics`):
```
modules/settings/
├── __init__.py    # Exporta el router
├── router.py      # Endpoints HTTP
└── service.py     # Lógica + acceso a DB inline (cuando es trivial)
```

**Módulo medio** (`cleanup`, `duplicates`):
```
modules/cleanup/
├── __init__.py
├── router.py
├── service.py
├── repository.py
└── schemas.py
```

**Módulo complejo** (`pipeline`, `library`):
```
modules/pipeline/
├── __init__.py
├── router.py            # Endpoints HTTP
├── service.py           # Orquestación: ejecuta steps, gestiona estado
├── repository.py        # pipelines, pipeline_steps, history
├── schemas.py           # DTOs request/response
├── models.py            # PipelineInfo, StepInfo, StepStatus
├── steps/               # Un archivo por tipo de step
│   ├── chapters.py
│   ├── subtitles.py
│   ├── translate.py
│   └── dubbing.py
└── _internal/           # Helpers privados al módulo
    ├── eta.py
    └── diff_detector.py
```

### 3.5. Las 6 reglas de oro (no negociables)

1. **Un router NUNCA llama a un client ni a la BD directamente.** Siempre pasa por su `service`.
2. **Un módulo solo expone su `router` desde `__init__.py`.** El resto es interno. Si otro módulo necesita lógica, importa el `service` o un `schema` público — nunca toca `repository` o `_internal`.
3. **`shared/` solo contiene utilidades sin estado y sin dominio.** Si algo crece y tiene reglas de negocio, se promueve a módulo propio.
4. **`clients/` está fuera de los módulos** porque varios módulos consumen los mismos backends. Si un cliente fuera exclusivo de un módulo, viviría dentro.
5. **Los módulos NO se importan entre sí circularmente.** Si dos módulos se necesitan mutuamente, falta extraer un tercero o subir lógica común a `shared/`.
6. **`main.py` solo hace registro.** Cero lógica. Cada módulo trae su router, lo registra, y listo.

### 3.6. Patrón canónico de cada capa

**`router.py`** — solo HTTP, validación y delegación:

```python
from fastapi import APIRouter, Depends
from .service import CleanupService
from .schemas import CleanupRequest, CleanupResponse

router = APIRouter(prefix="/api/cleanup", tags=["cleanup"])

@router.post("/scan", response_model=CleanupResponse)
async def scan(req: CleanupRequest, service: CleanupService = Depends()):
    # El router solo valida y delega. Cero lógica de negocio aquí.
    return await service.scan(req.path, dry_run=req.dry_run)
```

**`service.py`** — lógica de negocio pura:

```python
class CleanupService:
    """Servicio de limpieza: orquesta detección y borrado de artefactos huérfanos."""

    def __init__(self, repo: CleanupRepository, jobs: JobsService):
        self._repo = repo
        self._jobs = jobs

    async def scan(self, path: Path, *, dry_run: bool) -> CleanupResult:
        # Reglas de negocio. Sin saber de FastAPI ni de SQL directo.
        ...
```

**`repository.py`** — solo acceso a datos:

```python
class CleanupRepository:
    """Repositorio de limpieza: consultas SQL y acceso a ficheros."""

    def __init__(self, session: Session):
        self._session = session

    def find_orphan_srts(self, library_path: Path) -> list[OrphanSrt]:
        # SQL, ficheros, caché — sin lógica de negocio.
        ...
```

**`schemas.py`** — DTOs de entrada/salida HTTP (Pydantic):

```python
class CleanupRequest(BaseModel):
    path: str
    dry_run: bool = True

class CleanupResponse(BaseModel):
    orphans: list[OrphanItem]
    total_size_mb: float
```

**`models.py`** — entidades del dominio (dataclasses, cuando hace falta):

```python
@dataclass
class CleanupResult:
    orphans: list[OrphanItem]
    total_bytes: int
```

### 3.7. Inyección de dependencias

Para evitar singletons globales y facilitar tests, cada módulo declara sus dependencias con `Depends()` de FastAPI:

```python
# modules/cleanup/dependencies.py
def get_cleanup_service(
    session: Session = Depends(get_session),
    jobs: JobsService = Depends(get_jobs_service),
) -> CleanupService:
    return CleanupService(CleanupRepository(session), jobs)
```

**Ventaja en tests:** override trivial con `app.dependency_overrides[get_cleanup_service] = lambda: MockService()`.

### 3.8. Tests por módulo

```
tests/modules/cleanup/
├── test_router.py        # Tests con TestClient — golden path HTTP
├── test_service.py       # Tests unitarios de lógica
└── test_repository.py    # Tests con BD en memoria
```

Reemplaza la estructura plana actual de `tests/test_*.py`.

### 3.9. Estructura Layered — `ossflow-subtitle`, `ossflow-dubbing`, `ossflow-splitter`

```
ossflow-subtitle/
├── api/
│   └── router.py           # /run, /events, /gpu, /health
├── core/
│   ├── transcriber.py      # Wrapper WhisperX
│   ├── postprocessor.py    # Corrección gpt-4o, hotwords BJJ
│   ├── translator.py       # Ollama / OpenAI
│   └── srt_writer.py       # Serialización
├── shared/
│   └── paths.py
└── main.py
```

```
ossflow-dubbing/
├── api/
│   └── router.py
├── core/
│   ├── elevenlabs_client.py
│   ├── audio_aligner.py
│   └── muxer.py            # ffmpeg
├── shared/
└── main.py
```

```
ossflow-splitter/
├── api/
│   └── router.py
├── core/
│   ├── strategies/
│   │   ├── signal.py       # Detector ML
│   │   └── timestamps.py   # Corte por timestamps externos
│   └── splitter.py         # Orquestador (80% común)
├── shared/
└── main.py
```

### 3.10. Estructura Vertical Slice ligero — `ossflow-telegram`

```
ossflow-telegram/
├── modules/
│   ├── channels/      # CRUD de canales
│   ├── downloads/     # Jobs de descarga de media
│   └── auth/          # Autenticación Telegram
├── shared/
└── main.py
```

### 3.11. Estructura Vertical Slice — `ossflow-scrapper`

```
ossflow-scrapper/
├── modules/
│   ├── search/        # Búsqueda cross-provider
│   ├── scrape/        # Endpoint principal de scraping
│   └── providers/
│       ├── bjjfanatics/        # Provider actual
│       ├── grappling_industries/  # Fase 2
│       └── youtube/             # Fase 2
├── shared/
└── main.py
```

---

## 4. Estrategia de Migración

### Decisión: **Big Bang**

Se ejecutará el refactor completo en una única iteración, congelando el desarrollo de features durante el periodo necesario.

**Justificación:**

- El proyecto es personal y la mayoría de la funcionalidad está terminada y estable
- El autor controla totalmente el ritmo de desarrollo
- Big bang produce un resultado **coherente desde el primer commit** sin arquitecturas mezcladas conviviendo temporalmente
- Evita la complejidad operativa de mantener código viejo y nuevo en paralelo (strangler fig)
- Reduce el riesgo total: una sola pasada de validación al final

**Trade-off aceptado:** desarrollo de features pausado durante el refactor.

### Fases del big bang

**Fase 1 — Preparación**
- Crear los 4 repositorios vacíos en GitHub (`ossflow-core`, `ossflow-platform`, `ossflow-scrapper`, `ossflow-studio`)
- Tag de seguridad sobre el monorepo actual antes de tocar nada (`v-pre-refactor`)
- Backup de la base de datos SQLite (`/data/db/bjj.db`)

**Fase 2 — Split de código**
- Mover el código a los 4 repos sin refactorizar internamente
- Renombrar `bjj_*` → `ossflow_*` en imports, paquetes, identificadores Docker
- Verificar arranque completo y tests pasando

**Fase 3 — Refactor interno por servicio**
- Aplicar Vertical Slice / Layered según la decisión por servicio
- Empezar por `ossflow-api` (mayor caos, mayor beneficio)
- Continuar con `ossflow-scrapper`, luego servicios Layered
- Migrar tests al nuevo layout `tests/modules/<modulo>/`

**Fase 4 — Validación end-to-end**
- Smoke test de pipeline completo con un instruccional real
- Verificar todos los endpoints existentes siguen respondiendo igual
- Verificar SSE, ring buffer logs, GPU proxying

**Fase 5 — Reanudar desarrollo**

El **plan detallado de implementación** se elaborará en un documento aparte (siguiente paso tras aprobar este diseño).

---

## 5. Distribución de `ossflow-service-kit`

### Decisión: **Git tag + pip install desde GitHub**

`ossflow-service-kit` se distribuye como dependencia Python instalable directamente desde el repositorio `ossflow-core` mediante tags de versión.

**Sintaxis en el `pyproject.toml` de cada servicio:**

```toml
dependencies = [
    "ossflow-service-kit @ git+https://github.com/<usuario>/ossflow-core@v1.0.0#subdirectory=ossflow_service_kit",
]
```

**Justificación:**

- Cero coste de infraestructura
- Versionado real con tags semánticos (`v1.0.0`, `v1.1.0`, ...)
- En CI: GitHub deploy keys o personal access token vía secrets
- Si el proyecto madura y necesita PyPI privado, la migración es trivial: cambiar la URL en cada `pyproject.toml`

**Trade-off aceptado:** ligeramente más lento que un PyPI privado (pip clona el repo); irrelevante a esta escala.

---

## 6. Convenciones Globales

- **Idioma:** todos los comentarios, docstrings y documentación en castellano
- **Tipado:** Python 3.11+ con type hints en todas las firmas públicas
- **Estilo:** Pydantic v2 para schemas, `dataclass` para entidades de dominio internas, `from __future__ import annotations` activado
- **Imports:** absolutos siempre (`from ossflow_api.modules.pipeline.service import ...`); nunca imports relativos cruzando módulos
- **Tests:** pytest con `tests/modules/<modulo>/` por módulo, fixtures compartidas en `tests/conftest.py`
- **Docker:** código en imagen, no bind-mount (regla heredada que sigue vigente). Cambios requieren `docker compose build && docker compose up -d`

---

## 7. Resumen Ejecutivo

| Decisión | Valor |
|---|---|
| Lenguaje principal | Python (mantener) |
| Repositorios | 4: `ossflow-core`, `ossflow-platform`, `ossflow-scrapper`, `ossflow-studio` |
| Patrón en `ossflow-api` | Vertical Slice (~18 módulos) |
| Patrón en `ossflow-scrapper` | Vertical Slice (providers) |
| Patrón en `ossflow-telegram` | Vertical Slice ligero |
| Patrón en `ossflow-subtitle` | Layered |
| Patrón en `ossflow-dubbing` | Layered |
| Patrón en `ossflow-splitter` | Layered + Strategy |
| Contenido de `ossflow-service-kit` | Solo infraestructura transversal |
| Estrategia de migración | Big bang |
| Distribución del kit compartido | Git tag + pip install desde GitHub |
| Idioma de comentarios y docs | Castellano |

---

## 8. Próximos Pasos

1. Revisión y aprobación de este diseño por el autor
2. Elaboración del plan de implementación detallado (fase por fase, módulo por módulo)
3. Ejecución del big bang siguiendo el plan
