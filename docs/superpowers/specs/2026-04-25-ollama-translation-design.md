# Ollama local translation — design spec

**Fecha:** 2026-04-25
**Autor:** Claude (Opus 4.7) + adrianns93
**Estado:** v3 — incorpora 1ª y 2ª revisión arquitectónica (aprobado con ajustes menores)

---

## 1. Objetivo

Reemplazar la traducción EN→ES de OpenAI (`gpt-4o`) por un LLM **local gratuito** servido por Ollama (`qwen2.5:7b-instruct-q4_K_M`), conservando la calidad de adaptación BJJ del `OpenAITranslator` actual (terminología grip/hook, castellano peninsular, modo dubbing iso-síncrono con char budget y feedback retries).

El sistema queda con dos providers reales: **Ollama (default)** y **OpenAI (fallback opt-in)**. **DeepL se elimina por completo**.

## 2. Motivación y contexto

- Ejecutar todo el pipeline en OpenAI sale caro: ~$22 cubren ~3 seasons de instruccionales BJJ.
- DeepL no es opción real — no soporta el modo `dubbing_mode` (no respeta char budget per-slot) y la rama existente solo se usaba como fallback teórico.
- La 2070 Super (8 GB VRAM) y 32 GB RAM permiten correr Qwen2.5-7B Q4 (~4.5 GB VRAM) **secuencialmente** junto a WhisperX/demucs/Kokoro: el usuario nunca corre stages en paralelo.
- El `OpenAITranslator` actual ya acepta `base_url` parametrizable (`subtitle-generator/subtitle_generator/translator.py:386`), pero las pruebas reales con qwen2.5 muestran que `response_format=json_object` no es 100% fiable vía `/v1/chat/completions` (devuelve markdown fences ocasionalmente). Por eso este spec introduce un **`OllamaTranslator` dedicado** que usa el endpoint nativo `/api/chat` con `format=json` estricto. Para evitar duplicación de código se extrae primero un `_BaseChatTranslator` que comparte retries, prompts y feedback loops entre ambos providers (sección 4.2).

## 3. Decisiones de alcance

| Pregunta | Decisión |
|---|---|
| ¿Qué translator se migra? | **Solo `subtitle-generator/subtitle_generator/translator.py`**. El `dubbing-generator/dubbing_generator/translation/translator.py` (Helsinki-NLP MarianMT) es código muerto y se **borra**. |
| ¿Qué pasa con DeepL? | **Eliminado completamente** del repo. Ya no es fallback ni provider. |
| ¿Default tras la migración? | **Ollama** (`translation_provider="ollama"`). OpenAI queda como fallback opt-in. **Mitigación del riesgo de calidad:** el frontend deja un dropdown muy visible para volver a OpenAI con un click. |
| ¿Endpoint Ollama? | **Nativo `/api/chat` con `format="json"`** (NO `/v1/chat/completions`). El arquitecto detectó que `response_format=json_object` no es estricto en qwen2.5 vía la API OpenAI-compat: a veces devuelve `\`\`\`json…\`\`\`` o "Here is the JSON:" antes del objeto. El endpoint nativo es estricto. |
| ¿Cómo se descarga el modelo Qwen? | **Pull automático en `entrypoint.sh`** del contenedor Ollama. Primer arranque tarda ~5-10 min (~4.5 GB), después cacheado en volumen Docker. |
| ¿Modelo concreto? | `qwen2.5:7b-instruct-q4_K_M` — mejor multilingüe ES/EN del rango 7B, ES peninsular sin pelearse con el prompt, soporta `format=json` estricto, ~4.5 GB VRAM. |
| ¿`OLLAMA_KEEP_ALIVE`? | **`2m`** (no 30m). Tras la fase de traducción la VRAM se libera rápido para que WhisperX/demucs/Kokoro no queden bloqueados en pipeline secuencial. La penalización de reload (~5s) es despreciable. |

**Fuera de alcance:**
- Migrar otros consumidores de OpenAI (subtitle postprocess `gpt-4o-mini` se queda en OpenAI por ahora).
- Soportar múltiples modelos Ollama simultáneamente.
- Optimizar el char budget para la latencia distinta de Ollama vs OpenAI (los retries existentes absorben desvíos).

## 4. Arquitectura

```
┌────────────────────────────────────────────────────────────────────┐
│                      Docker compose network                        │
│                                                                    │
│  ┌────────────────┐                                                │
│  │ subtitle-      │  POST /api/chat  (format=json)                 │
│  │ generator      │ ───────────────────► ┌──────────────────────┐  │
│  │ (translator.py:│                      │  ollama              │  │
│  │  OllamaTrans-  │                      │  (ollama/ollama:     │  │
│  │  lator)        │                      │   latest)            │  │
│  └────────────────┘                      │  - GPU passthrough   │  │
│         ▲                                │  - volumen modelos   │  │
│         │ /translate                     │  - entrypoint pull   │  │
│         │                                │  - KEEP_ALIVE=2m     │  │
│  ┌────────────────┐                      └──────────────────────┘  │
│  │ processor-api  │                                                │
│  │ (pipeline.py,  │                                                │
│  │  subtitles.py) │                                                │
│  └────────────────┘                                                │
│         ▲                                                          │
│         │ PUT /api/settings                                        │
│         │ (translation_provider="ollama"|"openai")                 │
│  ┌────────────────┐                                                │
│  │ processor-     │                                                │
│  │ frontend       │                                                │
│  │ (SettingsPage) │                                                │
│  └────────────────┘                                                │
└────────────────────────────────────────────────────────────────────┘
```

### 4.1 Servicio Ollama (nuevo)

**`docker-compose.yml`** añade:

```yaml
ollama:
  image: ollama/ollama:latest
  container_name: bjj-ollama
  restart: unless-stopped
  networks:
    - bjj_net
  ports:
    - "11434:11434"  # opcional, solo para debug; los servicios la usan vía red interna
  volumes:
    - ollama-models:/root/.ollama
    - ./ollama/entrypoint.sh:/entrypoint.sh:ro
  environment:
    - OLLAMA_KEEP_ALIVE=2m
    - OLLAMA_HOST=0.0.0.0:11434
  entrypoint: ["/bin/bash", "/entrypoint.sh"]
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
    start_period: 1200s  # 20 min para el primer pull (CDN saturado / red lenta)
```

Top-level `volumes:` añade `ollama-models:`.

**Decisión `depends_on`:** los servicios `subtitle-generator` y `processor-api` **NO** declaran `depends_on: ollama:service_healthy`. Esto evita bloquear el bootstrap del frontend durante los ~10 minutos del primer pull. Si una llamada de traducción ocurre antes de que Ollama esté listo, el `OllamaTranslator` lanzará `httpx.ConnectError` → cae al fallback OpenAI si está configurado, si no → HTTP 502 con mensaje claro al usuario ("Ollama todavía está descargando el modelo, espere unos minutos").

**`ollama/entrypoint.sh`** (nuevo):

```bash
#!/bin/bash
set -e

# Arrancar Ollama en background
/bin/ollama serve &
OLLAMA_PID=$!

# Esperar a que el servidor responda
until curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; do
  sleep 2
done

# Pull idempotente del modelo
MODEL="qwen2.5:7b-instruct-q4_K_M"
if ! curl -fsS http://localhost:11434/api/tags | grep -q "$MODEL"; then
  echo "Pulling $MODEL (primer arranque, ~4.5 GB)..."
  ollama pull "$MODEL"
fi

# Devolver control al proceso ollama
wait $OLLAMA_PID
```

**Razones de diseño:**
- Pull en entrypoint en vez de `RUN` en Dockerfile: la imagen oficial es read-only para el modelo store; el modelo vive en volumen, no en la imagen.
- Healthcheck con `curl` cubre dos eventos en un solo comando (servidor up + modelo presente). Evita la race condition del `ollama list` (que requiere CLI Y que el server esté escuchando).
- `OLLAMA_KEEP_ALIVE=2m`: el modelo se descarga de VRAM rápido tras la fase de traducción, dejando la GPU libre para WhisperX/demucs/Kokoro en el siguiente paso del pipeline secuencial. La penalización de reload (~5s) es invisible al usuario.

### 4.2 Refactor a `_BaseChatTranslator` + nuevo `OllamaTranslator`

**Decisión técnica (D1):** en lugar de duplicar ~150 líneas entre `OpenAITranslator` y `OllamaTranslator`, se extrae primero un `_BaseChatTranslator` que comparte la lógica común (retry de count mismatch, retry de budget over/under, fallback one-by-one, prompts BJJ) y deja como abstractos los 4 puntos donde los providers difieren: URL, headers, shape del request body, shape de la respuesta. Esto alinea con el principio DRY de CLAUDE.md y evita divergencia futura cuando se tuneen prompts.

**`subtitle-generator/subtitle_generator/translator.py`** queda con:

- **Eliminado:** `DeepLTranslator`, constantes `DEEPL_FREE_URL`, `DEEPL_PRO_URL`, rama `deepl` en `make_translator`.
- **Refactor:** los métodos `_translate_batch` y `_translate_dubbing_batch` actualmente en `OpenAITranslator` se mueven a la nueva clase `_BaseChatTranslator(_BaseTranslator)`. Los retry loops, prompts y feedback messages quedan idénticos.
- **Métodos abstractos del base** (a implementar por cada provider concreto):
  - `_endpoint_url() -> str`
  - `_request_headers() -> dict`
  - `_wrap_chat_body(messages: list[dict], json_mode: bool) -> dict` — construye el body del POST según el dialecto del provider.
  - `_extract_message_content(response_json: dict) -> str` — saca el string de respuesta (en OpenAI: `response["choices"][0]["message"]["content"]`; en Ollama: `response["message"]["content"]`).
- **`OpenAITranslator(_BaseChatTranslator)`** queda reducido a esos 4 métodos + `__init__`. Mantiene la API pública (`translate_texts`, `translate_for_dubbing`, `translate_srt`).
- **`OllamaTranslator(_BaseChatTranslator)`** implementa los 4 métodos hablando con `/api/chat` nativo:

```python
class OllamaTranslator(_BaseChatTranslator):
    """Translate via Ollama native API (/api/chat with format=json strict)."""

    def __init__(
        self,
        model: str = "qwen2.5:7b-instruct-q4_K_M",
        source_lang: str = "EN",
        target_lang: str = "ES",
        temperature: float = 0.2,
        base_url: str | None = None,
    ) -> None:
        super().__init__(source_lang, target_lang)
        self.model = model
        self.temperature = temperature
        self.base_url = (base_url or os.environ.get(
            "OLLAMA_BASE_URL", "http://ollama:11434"
        )).rstrip("/")
        self.provider_label = "Ollama"

    def _endpoint_url(self) -> str:
        return f"{self.base_url}/api/chat"

    def _request_headers(self) -> dict:
        return {"Content-Type": "application/json"}

    def _wrap_chat_body(self, messages: list[dict], json_mode: bool) -> dict:
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if json_mode:
            body["format"] = "json"
        return body

    def _extract_message_content(self, resp_json: dict) -> str:
        return resp_json["message"]["content"]
```

**Diferencias técnicas Ollama vs OpenAI** (encapsuladas en los 4 métodos abstractos):

1. **Endpoint:** Ollama `POST {base_url}/api/chat`; OpenAI `POST {base_url}/chat/completions`.
2. **Body Ollama:**
   ```json
   {
     "model": "qwen2.5:7b-instruct-q4_K_M",
     "messages": [...],
     "format": "json",
     "stream": false,
     "options": { "temperature": 0.2 }
   }
   ```
3. **Sin Authorization header** en Ollama (no autentica en local).
4. **Respuesta Ollama:** `{"message": {"content": "<json string>"}, ...}` en vez de `{"choices":[{"message":{"content":...}}]}`.
5. **`format="json"` es estricto** en Ollama: valida que el output del modelo sea JSON válido o reintenta hasta serlo. Elimina el riesgo de markdown fences que tiene `/v1/chat/completions`.

**Lógica compartida vía `_BaseChatTranslator`:**
- `translate_texts` y `_translate_batch` (prompts `_BJJ_SYSTEM_PROMPT`, batch=40, retry de count mismatch con feedback message).
- `translate_for_dubbing` y `_translate_dubbing_batch` (prompts `_BJJ_DUBBING_*`, batch=15 si `fill_budget`, retry budget feedback con upper/lower bounds, fallback one-by-one).
- `_post_with_retry` ya parametriza `provider_label` (verificado en `subtitle-generator/subtitle_generator/translator.py:54-83`); cada subclase pasa el suyo.

**Beneficio del refactor:** ~120 líneas no duplicadas, riesgo cero de divergencia futura entre prompts/retries (los prompts BJJ se tunean con frecuencia — ver MEMORY.md sobre las 11 rondas de XTTS, patrón recurrente). Coste: ~1-2h de pasada de extracción + tests existentes deberían seguir verdes (lógica idéntica, solo movida).

**Factory `make_translator`:**

```python
def make_translator(
    provider: str,
    *,
    api_key: str | None = None,
    source_lang: str = "EN",
    target_lang: str = "ES",
    model: str | None = None,
    formality: str | None = None,
) -> _BaseTranslator:
    p = (provider or "").lower().strip()
    if p in ("ollama",):
        return OllamaTranslator(
            model=model or "qwen2.5:7b-instruct-q4_K_M",
            source_lang=source_lang,
            target_lang=target_lang,
        )
    if p in ("openai", "gpt", "chatgpt"):
        return OpenAITranslator(
            api_key=api_key,
            model=model or "gpt-4o-mini",
            source_lang=source_lang,
            target_lang=target_lang,
        )
    raise ValueError(f"Unknown translation provider: {provider!r}")
```

### 4.3 Refactor de `subtitle-generator/app.py`

`_build_translator_with_fallback` se simplifica:

- Eliminadas las 4 ocurrencias de `provider == "deepl"` (líneas 175-178, 194-197).
- Default de `provider` pasa de `"openai"` a `"ollama"`.
- Para `provider="ollama"` no se busca `api_key` en env (Ollama no la necesita).
- El error `"dubbing_mode requires OpenAI provider; DeepL has no budget-aware mode"` (línea 272) **se reescribe** a `"dubbing_mode requires a budget-aware translator (Ollama or OpenAI)"`. El `isinstance(primary, OpenAITranslator)` actual hay que **ampliar** a `isinstance(primary, (OpenAITranslator, OllamaTranslator))` ya que ambos implementan `translate_for_dubbing`.

**Mejora de fallback dubbing-aware** (recomendación arquitecto, NICE-TO-HAVE 12):
- En `_run_translate_directory` líneas 443-452 (rama dubbing_mode con fallback), hoy el fallback siempre llama `fallback.translate_srt(...)` modo literal. Cambio: si `fallback` es subclase de `_BaseChatTranslator` (cubre tanto `OpenAITranslator` como `OllamaTranslator`), intentar `_translate_for_dubbing(fallback, ...)` antes de caer a literal. Mantiene char budget cuando primary falla.

**Nota sobre el `isinstance` check existente:** `subtitle-generator/app.py:270` (en `_translate_for_dubbing`) hace `isinstance(primary, OpenAITranslator)` antes de llamar `translate_for_dubbing`. Tras el refactor del 4.2, ese check se amplía a `isinstance(primary, _BaseChatTranslator)` (cubre ambos providers en una sola comprobación). **Verificar con grep que no haya otras ocurrencias del isinstance en el módulo** durante la implementación.

### 4.4 Refactor de `processor-api/api/settings.py`

- `_DEFAULTS["translation_provider"]` cambia de `"openai"` a `"ollama"`.
- `_DEFAULTS["translation_fallback_provider"]` cambia de `"deepl"` a `"openai"`.
- `_DEFAULTS["translation_model"]` cambia de `"gpt-4o"` a `"qwen2.5:7b-instruct-q4_K_M"`.
- **Decisión de claves de modelo:** una sola clave `translation_model`. El frontend muestra placeholder/dropdown distinto según provider seleccionado.
- **Eliminación de DeepL en validación:** rama `if "deepl_api_key" in body` (líneas 222-226) **se elimina**. La key residual se purga en la migración (sección 8).
- **Validador de `translation_provider`:** acepta solo `("ollama", "openai")`. `"deepl"` o cualquier otro → 422.
- **Validador de `translation_fallback_provider`:** acepta `("", "ollama", "openai")` (string vacío = sin fallback, coherente con la convención existente).

### 4.5 Refactor de `processor-api/api/pipeline.py` y `subtitles.py`

**`processor-api/api/pipeline.py`** (líneas 369-422):
- Default del provider pasa a `"ollama"`.
- Rama de api_key:
  ```python
  if provider == "openai":
      api_key = get_setting("openai_api_key")
  elif provider == "ollama":
      api_key = None  # no necesita
  ```
- Rama deepl `else get_setting("deepl_api_key") if provider == "deepl"` **se elimina**.
- Idéntica lógica para `fallback`.

**`processor-api/api/subtitles.py`** (CRÍTICO — el arquitecto detectó que el spec v1 lo había omitido):
- Línea 137-139: rama `if provider == "deepl": return get_setting("deepl_api_key")` **se elimina**.
- Línea 144 (default `provider="openai"`): cambia a `"ollama"`.
- `_key_for_provider`: añadir rama `if provider == "ollama": return None` (Ollama no necesita key).
- Línea 165-167 (`raise HTTPException(... API key missing ...)`): el check debe excluir provider=ollama (no exigirle key). Adaptación:
  ```python
  if provider != "ollama" and not key:
      raise HTTPException(...)
  ```

### 4.6 Refactor de `processor-api/api/app.py`

Verificación rápida: si `processor-api/api/app.py` tiene wiring deepl en `run_translation` u otra función de orquestación (líneas ~614-657 según el arquitecto), eliminar esas ramas. Cambio quirúrgico, **una línea WIRE_TRANSLATOR_OLLAMA** en el plan para coordinar con cualquier feature paralela y evitar conflictos en este archivo central (CLAUDE.md convención).

### 4.7 Frontend `SettingsPage.jsx` (REWRITE COMPLETO de la sección Translation)

Este apartado fue subestimado en v1. El bloque DeepL son ~40 líneas de JSX (state `showDeepL`, badge Free/Pro, link a deepl.com, input para key, validación). El dropdown de modelo está hardcodeado a opciones OpenAI. Los cambios concretos:

- **Schema Zod:**
  - `translation_provider: z.enum(['ollama', 'openai'])` (era `['openai', 'deepl']`).
  - `translation_fallback_provider: z.enum(['', 'ollama', 'openai']).optional()` (mantener `''` como sentinel de "sin fallback", coherente con resto del archivo).
  - Eliminar `deepl_api_key` del schema y de `defaultValues`.

- **CardDescription:** reescribir literal "OpenAI (gpt-4o-mini) por defecto, DeepL como fallback" → "Ollama local (qwen2.5-7b) por defecto, OpenAI como fallback opcional".

- **Selector "Translation Provider":** dos opciones — `Ollama (local, gratis)` / `OpenAI (cloud, pago)`. Switch grande visible — el usuario lo verá fácilmente cuando quiera volver a OpenAI si la calidad de Ollama no convence.

- **Selector "Fallback Provider":** mismas dos opciones + opción `Ninguno` (que mapea a `''`).

- **Dropdown de modelo según provider:**
  - Si `provider === 'ollama'`: opciones `qwen2.5:7b-instruct-q4_K_M` (default) y `qwen2.5:14b-instruct-q4_K_M` (opcional, solo si el usuario tiene VRAM suficiente — nota tooltip).
  - Si `provider === 'openai'`: opciones existentes (`gpt-4o-mini`, `gpt-4o`, `gpt-4.1-mini`, `gpt-4.1`).
  - Implementación: condicional `{provider === 'ollama' ? <SelectOllama /> : <SelectOpenAI />}`.

- **Reset/clamp del modelo al cambiar provider** (NUEVO 4 de la 2ª revisión):
  - `useEffect` que escuche cambios en `translation_provider`. Si el `translation_model` actual no pertenece a la lista válida del nuevo provider (ej: `gpt-4o` quedó pegado y el usuario cambia a Ollama), se setea automáticamente al default del nuevo provider (`qwen2.5:7b-instruct-q4_K_M` o `gpt-4o-mini`). Evita que llegue al backend un combo inválido (Ollama + `gpt-4o` → 404 model not found en runtime).
  - ~5 líneas de JSX.

- **Eliminación completa del bloque DeepL JSX** (~40 líneas):
  - State `showDeepL`, `hasDeepL`, `deeplIsFree`, `deeplBadge`.
  - Input `deepl_api_key`, label, link a deepl.com, badge Free/Pro.
  - Cualquier `useEffect` que lea/escriba la key.

- **Banner informativo Ollama-down** (NUEVO 5 de la 2ª revisión, **incorporado en plan**):
  - Si `provider === 'ollama'` y backend reporta Ollama down (o pull en curso), banner amarillo: "Ollama está iniciando o descargando el modelo (~4.5 GB la primera vez). Mientras tanto puedes cambiar a OpenAI." con CTA al dropdown.
  - Infra reusada: `useBackendsHealth.js` polea cada 10s con react-query; basta con añadir `{ id: 'ollama', label: 'Ollama' }` al array `BACKENDS` y conectar el ping al endpoint `/api/health/ollama` (sección 10).
  - Coste real estimado: ~30 líneas distribuidas en 3 archivos (`useBackendsHealth.js`, backend `health_proxy.py`, `SettingsPage.jsx`).

### 4.8 Eliminaciones (código muerto)

**Borrar archivos:**
- `dubbing-generator/dubbing_generator/translation/translator.py` (Helsinki-NLP, sin uso confirmado).
- `dubbing-generator/dubbing_generator/translation/__init__.py` (si queda vacío tras borrar el translator; revisar contenido antes de borrar).
- `dubbing-generator/tests/test_translator.py` (test del translator borrado).

**Limpiar referencias a DeepL:**
- `subtitle-generator/subtitle_generator/translator.py`: clase `DeepLTranslator`, constantes `DEEPL_*`, factory branch.
- `subtitle-generator/app.py`: 4 ocurrencias del flujo deepl + reescritura del mensaje de error dubbing_mode.
- `processor-api/api/settings.py`: clave `deepl_api_key` y validador.
- `processor-api/api/pipeline.py`: rama provider deepl.
- `processor-api/api/subtitles.py`: rama deepl en `_key_for_provider`, default provider, check api_key.
- `processor-api/api/app.py`: si hay wiring deepl, eliminarlo (línea WIRE_TRANSLATOR_OLLAMA en plan).
- `processor-frontend/src/features/settings/pages/SettingsPage.jsx`: enum + ~40 líneas del bloque deepl + dropdown modelo.
- `dubbing-generator/_settings.json`: archivo residual (ya NO lo carga `processor-api/api/settings.py`, que usa SQLite). Confirmar con el usuario si se borra entero o se conserva como referencia histórica.
- `.env.example`: eliminar `DEEPL_API_KEY`.

**Dependencias:** `subtitle-generator/requirements.txt` no lista deepl (usa httpx puro) — nada que quitar ahí.

## 5. Flujo de datos

```
Usuario clic "Procesar" en frontend
        │
        ▼
processor-api: POST /api/jobs/.../translate (o run completo)
        │
        ▼
pipeline.py lee translation_provider del settings (SQLite)
  → si "ollama": api_key=None, model=qwen2.5:7b-instruct-q4_K_M
  → si "openai": api_key=settings.openai_api_key, model=gpt-4o
        │
        ▼ proxy via JobEvent stream
subtitle-generator: _build_translator_with_fallback
  → make_translator("ollama", ...) → OllamaTranslator(base_url="http://ollama:11434")
        │
        ▼
OllamaTranslator.translate_for_dubbing(items, cps, fill_budget=True)
  → POST http://ollama:11434/api/chat
  → format="json" estricto, prompts BJJ idénticos a OpenAI
  → Ollama carga qwen2.5:7b en VRAM (~5s primer call, cached 2 min)
  → respuesta JSON {"t":[...]} con misma cantidad de items (garantizado por format=json)
  → retry budget feedback hasta 3 veces si over/undershoot
        │
        ▼
.es.srt escrito en disco. Tras el step, /api/pipeline/flush-ollama libera VRAM.
        │
        ▼
Siguiente step (dubbing/Kokoro) tiene la GPU libre.
```

## 6. Manejo de errores

| Escenario | Comportamiento esperado |
|---|---|
| Ollama no arrancó / pull en curso | `OllamaTranslator` lanza `httpx.ConnectError` → fallback a OpenAI si está configurado, si no → HTTP 502 con mensaje "Ollama todavía no está listo, espera unos minutos o cambia a OpenAI en Settings". |
| Qwen alucina cualquier formato | `format="json"` lo previene a nivel Ollama. Si aún así falla parse, retry loop (idéntico al de OpenAI) maneja el error. |
| Qwen devuelve N items ≠ input | Retry feedback loop (copiado de OpenAI, líneas 587-607). Tras 3 intentos cae a one-by-one. |
| One-by-one también falla | `RuntimeError` propaga a `_run_translate_directory`. Si fallback configurado: ejecuta fallback (dubbing-aware si fallback es OpenAI, literal si no). Si no hay fallback: error al frontend. |
| OOM en Ollama (Qwen + WhisperX simultáneos) | Pipeline secuencial mitiga. `KEEP_ALIVE=2m` libera VRAM rápido. Si aún ocurre: `subtitle-generator` recibe HTTP 500, fallback a OpenAI. |
| DeepL key residual en SQLite | Auto-migración (sección 8) la purga al primer arranque. |
| Ollama responde JSON malformado tras retries | One-by-one re-intenta. Si también falla, fallback OpenAI ejecuta en modo dubbing-aware (mejora del spec v2 §4.3) o literal si no hay budget-aware fallback. |

## 7. Testing

### 7.1 Tests unitarios nuevos (mockeados, NO requieren Ollama corriendo)

**`subtitle-generator/tests/test_translator_ollama.py`:**
- `make_translator("ollama")` devuelve `OllamaTranslator` con base_url default `http://ollama:11434`.
- Si `OLLAMA_BASE_URL` está en env, lo respeta.
- `model=None` resuelve a `qwen2.5:7b-instruct-q4_K_M`.
- `OllamaTranslator.translate_texts(["hello"])` con httpx mockeado: verifica que el body POST tiene `format: "json"`, `stream: false`, `messages: [...]` y que parsea correctamente `{"message":{"content":"{\"t\":[\"hola\"]}"}}`.
- `translate_for_dubbing` con fill_budget=True: verifica que retry feedback se ejecuta cuando algún ítem queda fuera de budget (mock devuelve item under-budget en primer intento, in-budget en segundo).
- Test de count mismatch retry: mock devuelve N-1 ítems → retry → devuelve N → pasa.

**`subtitle-generator/tests/test_translator_base_extraction.py`** (regresión post-refactor):
- Verifica que `OpenAITranslator` mantiene comportamiento idéntico antes/después de extraer `_BaseChatTranslator` (los tests existentes de OpenAITranslator deben seguir pasando sin modificación).
- Verifica que ambas subclases (`OpenAITranslator`, `OllamaTranslator`) implementan los 4 métodos abstractos requeridos.

### 7.2 Test automatizable de calidad de traducción (golden set)

**`subtitle-generator/tests/test_translator_quality.py`** + `subtitle-generator/tests/data/bjj_golden_set.jsonl`:

- **Golden set:** ~20 frases reales BJJ EN→ES, con traducción esperada peninsular y glosario respetado:
  ```jsonl
  {"en": "Get your grip on the sleeve and pass the guard.", "must_keep_en": ["grip", "guard"], "expected_register": "peninsular"}
  {"en": "Use the butterfly hook to sweep him.", "must_keep_en": ["butterfly hook", "sweep"], "expected_register": "peninsular"}
  ...
  ```
- **Métricas computadas** sobre el output de `OllamaTranslator.translate_for_dubbing`:
  - **(a) Glosario respetado:** % de ítems donde TODOS los `must_keep_en` aparecen literales en la traducción ES.
  - **(b) Castellano peninsular:**
    - **Señales NEGATIVAS** (latinismos prohibidos, regex case-insensitive): `\b(ahorita|ustedes|agarrar|agarra|agarro|agarras|chévere|chido|párate|parate|nomás|porfa)\b`. Cualquier match → ítem falla.
    - **Señales POSITIVAS** (esperadas en castellano peninsular coach BJJ): el % de ítems que contengan al menos una de `\b(coge|coges|fíjate|vale|tú|tus|hostia|joder|venga|aquí|ahora|mira)\b` debe ser ≥60%. Frecuencias bajas indican un registro neutro/latam que no cumple el prompt.
    - Métrica final = % de ítems que pasan AMBAS comprobaciones.
  - **(c) Char budget compliance:** % de ítems en rango [80%, 110%] del budget cuando `fill_budget=True`.
- **Umbrales** (configurables en el test):
  - Pasa: las 3 métricas ≥ 85%.
  - Falla: cualquier métrica < 70%.
  - Warning (no falla CI): métricas en rango [70%, 85%].
- **Marcado:** `@pytest.mark.quality` y `@pytest.mark.requires_ollama`. NO corre por defecto (`pytest -m "not requires_ollama"` lo salta). Se ejecuta solo en CI específico de calidad o on-demand. Su valor real es bloquear regresiones cuando se actualicen prompts BJJ o se cambie el modelo Qwen.
- **Registro de markers obligatorio** (corrección 3ª revisión): añadir a `subtitle-generator/pyproject.toml` en `[tool.pytest.ini_options]`:
  ```toml
  markers = [
    "requires_ollama: requiere servicio Ollama corriendo en localhost:11434",
    "quality: test de calidad sobre golden set BJJ",
  ]
  ```
  Sin este registro pytest emite `PytestUnknownMarkWarning` y el filtro `-m "not requires_ollama"` no funciona.
- **Prerrequisito de ejecución:** el host pytest llega a Ollama vía `localhost:11434` (puerto publicado en compose). Documentar en el docstring del test.

**Justificación:** la mitigación principal del riesgo "calidad < OpenAI" del spec es smoke test A/B manual. Sin un test automatizable, esa mitigación se evapora cuando el usuario en el futuro tunee un prompt o suba a Qwen2.5-14B. Este golden set establece una baseline reproducible.

### 7.3 Tests unitarios eliminados

- `subtitle-generator/tests/test_translator.py` (si existe): eliminar tests de `DeepLTranslator`. Conservar tests de `OpenAITranslator` (deben seguir pasando tras el refactor a `_BaseChatTranslator`).
- `dubbing-generator/tests/test_translator.py`: borrar entero.

### 7.4 Smoke test manual obligatorio antes de marcar la migración como done

1. `docker compose up -d ollama` → esperar healthcheck verde (`docker compose ps`, status healthy). Tarda ~10 min primera vez.
2. `curl http://localhost:11434/api/tags` debe listar `qwen2.5:7b-instruct-q4_K_M`.
3. `docker compose up -d` (resto de servicios).
4. `docker compose exec subtitle-generator python -c "from subtitle_generator.translator import make_translator; t = make_translator('ollama'); print(t.translate_texts(['Get your grip on the sleeve and pass the guard.']))"` → debe responder con ES peninsular conservando "grip" y "guard" en inglés.
5. **Smoke test E2E:** procesar 1 episodio real con `provider=ollama` + `dubbing_mode=true`. Verificar:
   - `.es.srt` generado conserva grip/hook/underhook.
   - Char budget respetado (mirar líneas vs duración slot).
   - Castellano peninsular ("tú", "coge", "vale", "fíjate").
   - No "agarrar" (latam), "ahorita", "ustedes".
6. **Comparación A/B (recomendada por arquitecto):** mismo episodio con `provider=openai` (gpt-4o), diff visual de los .es.srt. Si calidad Ollama es notablemente peor en >20% de líneas, considerar bajar el default a OpenAI antes de cerrar la migración.
7. Verificar que tras la fase translate, la GPU se libera en ~2 minutos (`nvidia-smi` muestra `qwen2.5` desaparecido) y que el siguiente step (Kokoro dubbing) arranca sin OOM.

## 8. Migración / rollout

**Storage real:** `processor-api` usa SQLite (`bjj_service_kit.db`) desde la migración previa, NO `_settings.json`. La migración corre contra la SQLite.

**Auto-migración del `Setting` rows** (en `processor-api/api/settings.py`, dentro de `_ensure_initialized()` tras `init_db()`, antes de `_maybe_import_legacy_json`).

**Importante (corrección de v2 tras 2ª revisión):** los valores en `Setting.value` están serializados como JSON (`save_settings` hace `json.dumps(v, ensure_ascii=False)` antes de escribir). Por lo tanto la migración usa la API real de SQLAlchemy + json, NO helpers ficticios:

```python
import json
from bjj_service_kit.db import session_scope
from bjj_service_kit.db.models import Setting

def _migrate_legacy_translation_settings() -> None:
    """One-shot rewrite de valores legacy. Idempotente."""
    with session_scope() as s:
        # 1. provider deepl → ollama (+ model default coherente)
        row = s.get(Setting, "translation_provider")
        if row is not None and json.loads(row.value) == "deepl":
            row.value = json.dumps("ollama")
            model_row = s.get(Setting, "translation_model")
            if model_row is not None:
                model_row.value = json.dumps("qwen2.5:7b-instruct-q4_K_M")

        # 2. fallback deepl → openai
        fb_row = s.get(Setting, "translation_fallback_provider")
        if fb_row is not None and json.loads(fb_row.value) == "deepl":
            fb_row.value = json.dumps("openai")

        # 3. purge deepl_api_key (clave entera, no solo el valor)
        deepl_row = s.get(Setting, "deepl_api_key")
        if deepl_row is not None:
            s.delete(deepl_row)
```

**Verificación de nombres reales** (validados contra el repo en 3ª revisión arquitectónica): `Setting` está en `bjj_service_kit.db.models` (NO en `api/settings.py`), `session_scope` está en `bjj_service_kit.db`. Modelo SQLAlchemy con primary key `key` (string) y columna `value` (string JSON serializado). NO existen helpers `_read_setting`/`_write_setting`/`_delete_setting` — usar la API directa de la sesión como muestra el snippet.

**Propiedades:**
- Idempotente: si los valores ya están migrados, los `if json.loads(...) == "deepl"` no disparan reescritura.
- Si la SQLite no existe: `init_db()` la crea; las rows simplemente no estarán presentes (`s.get(...)` devuelve None) y `load_settings()` aplica los nuevos defaults. Sin operación.
- Si el usuario ya tiene `translation_provider="openai"`: **se respeta** (la condición `== "deepl"` no se cumple). Solo migra deepl→ollama, no openai→ollama.

**`dubbing-generator/_settings.json`:** archivo residual (probablemente de una versión anterior pre-SQLite). Confirmar con el usuario si se borra el archivo. Si se conserva, la `deepl_api_key` que contiene queda inerte.

**Sin migración de datos de archivos:** los `.es.srt` ya generados con OpenAI quedan tal cual.

**Plan B explícito:** si Qwen2.5-7B da resultados peores tras smoke test A/B, el usuario cambia el dropdown del frontend a OpenAI y guarda. El `translation_provider="openai"` queda persistido en SQLite y prevalece. **Cero redeploy.**

## 9. Riesgos

| Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|
| Qwen2.5-7B no respeta peninsular tan bien como GPT-4o | **Alta** | Calidad doblaje | Prompts ya tienen reglas explícitas. Test golden set automatizable (sección 7.2) + smoke test A/B obligatorio (sección 7.4). Switch fácil a OpenAI desde frontend. |
| Qwen2.5-7B Q4 pierde adherencia en prompts 60+ líneas | **Media-Alta** | Char budget, count mismatches | Retry feedback loop existente (heredado de OpenAI). Fallback a one-by-one. Si crítico, subir a Q5_K_M (5.4 GB) o Qwen2.5-14B Q4 (~9 GB, no entra). |
| `format="json"` estricto rechaza algunos outputs | Baja | Latencia | Retry loop maneja. format=json es más fiable que markdown-fence-strip. |
| Char budget retries explotan (3 intentos sin converger) | Media | Calidad | One-by-one fallback. Si pasa mucho, bajar batch_size de 15 a 8 para fill_budget. |
| Pull del modelo falla en primer arranque (CDN, red) | Baja | UX bootstrap | Healthcheck en compose lo detecta. El usuario puede `docker compose exec ollama ollama pull qwen2.5:7b-instruct-q4_K_M` manual. |
| Ollama crashea durante un job largo | Baja | Job perdido | `restart: unless-stopped`. Job actual falla, fallback OpenAI o reintento manual. |
| VRAM bloqueada por KEEP_ALIVE | Baja | OOM en Kokoro/WhisperX | `KEEP_ALIVE=2m` mitiga. Endpoint `/flush-ollama` (sección 10) como kill-switch. |
| `_settings.json` legacy del dubbing-generator confunde al usuario | Baja | UX | Confirmar con usuario si borrarlo. |
| Migración SQLite no idempotente | Baja | Bucle de reescritura | El diseño es idempotente por construcción (solo reescribe si valor == "deepl"). |

## 10. Observabilidad

- **Logs `OllamaTranslator`:** mismo formato que `OpenAITranslator`, pero con `provider_label="Ollama"` parametrizable en `_post_with_retry` (cambio chico, beneficia ambas clases).
- **Healthcheck Ollama:** visible en `docker compose ps`.
- **Health de Ollama vía endpoint agregado existente** (decisión 3ª revisión): NO se añade `/api/health/ollama` dedicado. En su lugar, `processor-api/api/health_proxy.py` extiende el endpoint existente `/health/backends` añadiendo una entrada `ollama` al payload agregado (proxy a `GET http://ollama:11434/api/tags` internamente). El frontend ya consume `/health/backends` desde `useBackendsHealth.js` con react-query — basta con añadir `{ id: 'ollama', label: 'Ollama' }` al array `BACKENDS`. Coherente con la convención del repo (un solo endpoint agregado, no N endpoints dedicados por servicio).
- **Endpoint nuevo `/api/pipeline/flush-ollama`** en `processor-api/api/pipeline.py` (análogo al `/flush-gpu` existente): descarga el modelo de VRAM al instante mediante una llamada a Ollama. Implementación recomendada (más limpia que `/api/generate`):
  ```python
  POST http://ollama:11434/api/chat
  {
    "model": "<translation_model setting>",
    "messages": [],
    "keep_alive": 0,
    "stream": false
  }
  ```
  El `model` es **obligatorio** en este endpoint (sin él Ollama devuelve 400). Se lee del setting `translation_model` para no hardcodear. Útil tras la fase translate antes de Kokoro.

## 11. Plan de implementación (alto nivel — el plan detallado lo escribe writing-plans)

**Reordenado tras 2ª revisión** para que los unit tests del translator (mockeados) corran antes de levantar infra Ollama. Pasos backend disjuntos paralelizables explicitados.

1. **Refactor base:** extraer `_BaseChatTranslator` desde `OpenAITranslator` en `subtitle-generator/subtitle_generator/translator.py`. Mover `_translate_batch`, `_translate_dubbing_batch`, prompts BJJ. Definir 4 métodos abstractos. **Tests existentes de OpenAITranslator deben seguir verdes sin modificación.**
2. **Nuevo translator:** crear `OllamaTranslator(_BaseChatTranslator)`. Eliminar `DeepLTranslator`. Ampliar `make_translator` con rama ollama. Eliminar rama deepl.
3. **Tests translator (mockeados, no requieren infra):** `test_translator_ollama.py`, `test_translator_base_extraction.py`, eliminar tests deepl. Paso 3 valida pasos 1+2 antes de tocar infra.
4. **Infra Ollama:** crear `ollama/entrypoint.sh`, añadir servicio + volumen al `docker-compose.yml`. NO añadir `depends_on`. Healthcheck con `curl /api/tags | grep`.
5. **subtitle-generator app:** refactor `app.py` (default=ollama, eliminar deepl, ampliar `isinstance` check a `_BaseChatTranslator`, mejorar fallback dubbing-aware).
6. **Pasos API backend disjuntos** (paralelizables — agentes separados, archivos no solapados):
   - **6a. settings.py:** defaults nuevos, validadores, eliminar deepl, añadir `_migrate_legacy_translation_settings` (con la API SQLAlchemy real, NO helpers ficticios).
   - **6b. pipeline.py:** defaults, eliminar deepl, endpoint nuevo `/api/pipeline/flush-ollama` (body con `model` obligatorio).
   - **6c. subtitles.py:** rama ollama en `_key_for_provider`, default provider, check api_key flexible.
7. **Wire central app.py + health_proxy.py:** archivo central, NO paralelizar. Cambios: (a) `api/app.py` eliminar ramas deepl si las hay, (b) `api/health_proxy.py` añadir entrada `ollama` al payload agregado de `/health/backends` (proxy a `GET http://ollama:11434/api/tags`). Implementador deja en el comentario del PR la línea `WIRE_TRANSLATOR_OLLAMA: api/app.py deepl branches removed; api/health_proxy.py /health/backends extended with ollama entry` para que writing-plans coordine con features paralelas (convención CLAUDE.md "no tocar app.py en agentes paralelos").
8. **Frontend split (NUEVO 8 de la 2ª revisión):**
   - **8a. SettingsPage core:** schema Zod, defaultValues, CardDescription, selectores provider/fallback, dropdown modelo condicional, eliminación bloque DeepL (~33 líneas), `useEffect` reset modelo al cambiar provider.
   - **8b. Banner Ollama-down + health pill:** añadir `{ id: 'ollama', label: 'Ollama' }` a `BACKENDS` en `useBackendsHealth.js` (consume `/health/backends` agregado, ya existente). Banner amarillo en SettingsPage cuando `provider=ollama && status=down`.
9. **Cleanup:** borrar `dubbing-generator/dubbing_generator/translation/`, `tests/test_translator.py`. Confirmar con usuario si borrar `dubbing-generator/_settings.json` residual. Eliminar `DEEPL_API_KEY` de `.env.example`.
10. **Test calidad golden set:** `tests/test_translator_quality.py` + `tests/data/bjj_golden_set.jsonl`. Marcado `@pytest.mark.requires_ollama`, NO corre por defecto.
11. **Build + smoke test final:** `docker compose build && up -d`, esperar pull (~10 min primera vez), smoke test E2E + A/B vs OpenAI sobre 1 episodio real. Verificar que VRAM se libera ~2 min tras último request.

**Dependencias entre pasos:**
- 1 → 2 → 3 (refactor base antes de nueva subclase, tests al final).
- 3 → 4 (tests verdes antes de infra).
- 4 → 11 (infra antes de smoke).
- 6a/6b/6c paralelizables tras 5.
- 7 secuencial tras 6 (toca app.py central).
- 8a → 8b (core antes de banner).
- 9, 10 paralelizables al final.

---

## Apéndice A — Por qué Qwen2.5-7B y no Llama-3.1-8B / Mistral-Nemo / Gemma-2-9B

| Modelo | VRAM Q4 | ES peninsular | format=json | Instruction-following prompts largos |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 4.5 GB | Bueno (corpus equilibrado) | Estricto | Sólido en prompts 60+ líneas |
| Llama-3.1-8B-Instruct | 5.0 GB | Tirando a latino | Estricto | Sólido pero hay que reforzar peninsular |
| Mistral-Nemo-12B | 7.5 GB | Muy bueno | Estricto | Excelente, pero deja sin VRAM para nada concurrente |
| Gemma-2-9B-it | 6.0 GB | Bueno | Estricto | Bueno, apurado de VRAM con WhisperX residual |

La elección es la mejor relación calidad/VRAM con margen para que WhisperX/Kokoro coexistan en la misma GPU si el usuario decide en el futuro paralelizar steps.

## Apéndice B — Variables de entorno nuevas

| Variable | Default | Dónde se lee |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://ollama:11434` | `subtitle-generator/translator.py` (`OllamaTranslator`) |
| `OLLAMA_KEEP_ALIVE` | `2m` | contenedor `ollama` |
| `OLLAMA_HOST` | `0.0.0.0:11434` | contenedor `ollama` |

`OPENAI_API_KEY`, `OPENAI_BASE_URL` se conservan para el fallback. `DEEPL_API_KEY` **se elimina** del `.env.example`.

## Apéndice C — Historial de cambios (v1 → v2 → v3)

### v1 → v2: incorporación 1ª revisión arquitectónica

1. **BLOCKER 1 resuelto:** sección 4.5 ampliada con refactor explícito de `processor-api/api/subtitles.py`.
2. **BLOCKER 2 resuelto:** sección 4.7 reescrita con detalle del frontend (dropdown modelo según provider, eliminación bloque DeepL).
3. **BLOCKER 4 resuelto:** decisión técnica — endpoint nativo `/api/chat` con `format=json`.
4. **IMPORTANTE 5 resuelto:** `KEEP_ALIVE=30m` → `2m`. Endpoint `/flush-ollama` añadido.
5. **IMPORTANTE 6 resuelto:** healthcheck reescrito con `curl /api/tags | grep`.
6. **IMPORTANTE 7 resuelto:** sección 8 reescrita contra SQLite.
7. **IMPORTANTE 8 resuelto:** `depends_on: ollama:service_healthy` eliminado.
8. **NICE 9 incorporado:** endpoint `/api/health/ollama`.
9. **NICE 10 incorporado:** `provider_label` parametrizable.
10. **NICE 12 incorporado:** fallback dubbing-aware.
11. **A1 confirmado por usuario:** default=Ollama mantenido + switch visible en frontend + smoke test A/B.

### v2 → v3: incorporación 2ª revisión arquitectónica

12. **NUEVO 1 (IMPORTANTE) resuelto:** sección 8 reescrita con la API real de SQLAlchemy (`session.get(Setting, key)`, `json.loads(row.value)`, `s.delete(...)`). NO se asumen helpers `_read_setting`/`_write_setting`/`_delete_setting` que no existían.
13. **NUEVO 2 (IMPORTANTE) resuelto:** body completo del `/flush-ollama` documentado con `model` obligatorio. Implementación recomendada: `POST /api/chat` con `messages: []` + `keep_alive: 0` (más limpio que `/api/generate`).
14. **NUEVO 3 (IMPORTANTE) resuelto — D1 confirmado por usuario:** refactor a `_BaseChatTranslator` incorporado. La duplicación de ~150 líneas se elimina extrayendo el cuerpo común y dejando 4 métodos abstractos (URL, headers, wrap body, extract content). Beneficio: cero divergencia futura entre prompts BJJ. Coste: 1-2h en paso 1 del plan.
15. **NUEVO 4 (NICE) incorporado:** `useEffect` reset/clamp del `translation_model` al cambiar provider en frontend. Evita combos inválidos (Ollama + `gpt-4o` → 404 model not found).
16. **NUEVO 5 (NICE) incorporado:** banner Ollama-down sale del estado "opcional" a parte del plan (paso 8b). 30 líneas reales en 3 archivos.
17. **NUEVO 6 (NICE) incorporado:** `isinstance(primary, OpenAITranslator)` se amplía a `isinstance(primary, _BaseChatTranslator)` (cubre ambos providers en una comprobación). Verificación grep durante implementación.
18. **NUEVO 7 (IMPORTANTE) resuelto — E1 confirmado por usuario:** test automatizable de calidad con `bjj_golden_set.jsonl` y métricas (glosario / peninsular / char budget). Marcado `@pytest.mark.requires_ollama`, NO corre por defecto. Paso 10 del plan.
19. **NUEVO 8 (NICE) incorporado:** paso 8 del plan dividido en 8a (core SettingsPage) y 8b (banner + health pill).
20. **NUEVO 9 (NICE) incorporado:** plan reordenado — translator (paso 1-3) ANTES de infra (paso 4) para que tests mockeados corran sin docker. Pasos 6a/6b/6c paralelizables explicitados.
21. **NUEVO 10 (NICE) resuelto:** la justificación de "duplicación controlada" se elimina porque el refactor (NUEVO 3) la evita. Coherencia DRY de CLAUDE.md preservada.
22. **NUEVO 11 (NICE) incorporado:** marker `WIRE_TRANSLATOR_OLLAMA` precisado en paso 7 con formato exacto a escribir en el comentario del PR.

### v3 ajustes inline (3ª revisión, sin pasar a v4)

23. Import correcto en snippet de auto-migración: `from bjj_service_kit.db import session_scope` + `from bjj_service_kit.db.models import Setting` (NO `from .settings import …`, que era circular y no existía).
24. Registro obligatorio de pytest markers (`requires_ollama`, `quality`) en `subtitle-generator/pyproject.toml` añadido como sub-paso del paso 10.
25. Métrica de "castellano peninsular" reforzada con señales positivas (`coge`, `fíjate`, `vale`, `tú`, `mira`…) además de las negativas (`agarrar`, `ahorita`, `ustedes`…).
26. `start_period` del healthcheck Ollama subido de 600s a 1200s (20 min) para tolerar CDN saturado o red lenta en el primer pull.
27. Decisión endpoint health: NO se añade `/api/health/ollama` dedicado. Se extiende `/health/backends` agregado existente — coherente con la convención del repo y con `useBackendsHealth.js`.
