# Ollama local translation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reemplazar la traducción EN→ES de OpenAI gpt-4o por Ollama qwen2.5-7b local en el subtitle-generator. Eliminar DeepL. Default Ollama, OpenAI fallback opt-in.

**Architecture:** Extraer un `_BaseChatTranslator` con la lógica común (retries, prompts BJJ, feedback loop). `OpenAITranslator` y nuevo `OllamaTranslator` heredan e implementan 4 métodos abstractos (URL, headers, body shape, response extract). Servicio `ollama` en docker-compose con pull automático en entrypoint, health agregado vía `/api/health/backends` existente.

**Tech Stack:** Python 3.12 + FastAPI + httpx (subtitle-generator, processor-api), SQLAlchemy + SQLite (settings), React 18 + Zustand + react-query (frontend), Docker compose con NVIDIA GPU passthrough, Ollama runtime con `qwen2.5:7b-instruct-q4_K_M`.

**Spec source:** `docs/superpowers/specs/2026-04-25-ollama-translation-design.md` (v3, aprobado).

---

## Pre-flight: branch + worktree

- [ ] **Step 0.1: Verify clean working tree on a dedicated branch**

```bash
git status --short
git checkout -b feat/ollama-translation
```

Expected: branch creado.

---

## Phase 1 — Refactor `_BaseChatTranslator` (TDD, sin tocar Ollama todavía)

### Task 1.1: Snapshot de comportamiento actual de OpenAITranslator

**Files:**
- Create: `subtitle-generator/tests/test_translator_openai_snapshot.py`

Este test fija el comportamiento actual con httpx mockeado. Sirve de safety net para el refactor: si la extracción del base rompe algo, este test falla.

- [ ] **Step 1.1.1: Write the snapshot test**

Escribe en `subtitle-generator/tests/test_translator_openai_snapshot.py`:

```python
"""Snapshot tests for OpenAITranslator pre-refactor.

Estos tests fijan el comportamiento ANTES de extraer _BaseChatTranslator.
Deben seguir verdes después del refactor sin modificación.
"""
from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest

from subtitle_generator.translator import OpenAITranslator


def _mock_openai_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [{"message": {"content": content}}]
    }
    return resp


def test_translate_texts_calls_openai_with_correct_body():
    t = OpenAITranslator(api_key="sk-test", model="gpt-4o-mini")
    fake_resp = _mock_openai_response('{"t":["hola mundo"]}')

    with patch("subtitle_generator.translator._post_with_retry", return_value=fake_resp) as m:
        out = t.translate_texts(["hello world"])

    assert out == ["hola mundo"]
    args, kwargs = m.call_args
    assert args[0] == "https://api.openai.com/v1/chat/completions"
    assert kwargs["headers"]["Authorization"] == "Bearer sk-test"
    body = kwargs["json_body"]
    assert body["model"] == "gpt-4o-mini"
    assert body["response_format"] == {"type": "json_object"}
    assert len(body["messages"]) == 2


def test_translate_for_dubbing_uses_budget_prompt():
    t = OpenAITranslator(api_key="sk-test")
    fake_resp = _mock_openai_response('{"t":["coge el grip"]}')

    items = [{"text": "Get your grip.", "duration_ms": 1500}]
    with patch("subtitle_generator.translator._post_with_retry", return_value=fake_resp) as m:
        out = t.translate_for_dubbing(items, cps=17.0)

    assert out == ["coge el grip"]
    body = m.call_args.kwargs["json_body"]
    system_prompt = body["messages"][0]["content"]
    assert "DUBBING" in system_prompt
    assert "max_chars" in system_prompt or "target_chars" in system_prompt


def test_translate_for_dubbing_fill_budget_uses_fill_prompt():
    t = OpenAITranslator(api_key="sk-test")
    fake_resp = _mock_openai_response('{"t":["coge el grip, fíjate"]}')

    items = [{"text": "Get your grip.", "duration_ms": 1500}]
    with patch("subtitle_generator.translator._post_with_retry", return_value=fake_resp) as m:
        t.translate_for_dubbing(items, cps=17.0, fill_budget=True)

    body = m.call_args.kwargs["json_body"]
    system_prompt = body["messages"][0]["content"]
    assert "target_chars" in system_prompt
    assert "FILL THE SLOT" in system_prompt
```

- [ ] **Step 1.1.2: Run the test to verify it passes against current code**

```bash
cd subtitle-generator && python -m pytest tests/test_translator_openai_snapshot.py -v
```

Expected: 3 tests PASS. Si falla, revisar el código actual de `OpenAITranslator` antes de seguir — el refactor debe preservar el comportamiento, así que el snapshot necesita pasar primero.

- [ ] **Step 1.1.3: Commit**

```bash
git add subtitle-generator/tests/test_translator_openai_snapshot.py
git commit -m "test(translator): snapshot OpenAITranslator behavior pre-refactor"
```

---

### Task 1.2: Extraer `_BaseChatTranslator`

**Files:**
- Modify: `subtitle-generator/subtitle_generator/translator.py`

Mover `_translate_batch`, `_translate_dubbing_batch`, `translate_texts`, `translate_for_dubbing` desde `OpenAITranslator` a una nueva clase abstracta. `OpenAITranslator` queda con 4 métodos pequeños.

- [ ] **Step 1.2.1: Add `_BaseChatTranslator` skeleton above `OpenAITranslator` in `translator.py`**

Inserta esta clase ENTRE `class _BaseTranslator` (existe ya) y `class OpenAITranslator`. NO toques `OpenAITranslator` todavía:

```python
class _BaseChatTranslator(_BaseTranslator):
    """Shared logic for chat-completion-style providers (OpenAI, Ollama).

    Subclases implementan 4 métodos abstractos para diferenciar dialecto HTTP.
    El cuerpo de retry/feedback/prompts queda compartido aquí.
    """

    _BATCH_SIZE = 40
    _MAX_RETRIES = 2

    # Subclases setean estos en __init__:
    model: str
    temperature: float
    provider_label: str

    # ---- API HTTP a implementar por subclases ----
    def _endpoint_url(self) -> str:
        raise NotImplementedError

    def _request_headers(self) -> dict:
        raise NotImplementedError

    def _wrap_chat_body(self, messages: list[dict], json_mode: bool) -> dict:
        raise NotImplementedError

    def _extract_message_content(self, resp_json: dict) -> str:
        raise NotImplementedError

    # ---- Lógica compartida (la mueves desde OpenAITranslator en step 1.2.3) ----
```

- [ ] **Step 1.2.2: Run the snapshot test — should still pass (no logic change yet)**

```bash
cd subtitle-generator && python -m pytest tests/test_translator_openai_snapshot.py -v
```

Expected: 3 PASS.

- [ ] **Step 1.2.3: Move `_translate_batch` and `_translate_dubbing_batch` from OpenAITranslator into `_BaseChatTranslator`**

Edit `translator.py`. En `OpenAITranslator._translate_batch`, reemplaza el cuerpo que construye `body = {...}` y llama `_post_with_retry` por las llamadas abstractas. Mueve el método entero al base. Cambios mínimos:

En `_translate_batch` (ahora en el base), reemplaza:

```python
body = {
    "model": self.model,
    "temperature": self.temperature,
    "response_format": {"type": "json_object"},
    "messages": [...],
}
headers = {
    "Authorization": f"Bearer {self.api_key}",
    "Content-Type": "application/json",
}
```

por:

```python
messages = [
    {"role": "system", "content": system},
    {
        "role": "user",
        "content": (
            "Translate each item in `items`. Return JSON "
            '{"t": [...]} with the same number of items in the same order.\n'
            + json.dumps(user_payload, ensure_ascii=False)
        ),
    },
]
body = self._wrap_chat_body(messages, json_mode=True)
headers = self._request_headers()
```

Y en la línea que extrae content:

```python
content = data["choices"][0]["message"]["content"]
```

por:

```python
content = self._extract_message_content(data)
```

Y en la llamada `_post_with_retry`:

```python
r = _post_with_retry(
    self.url, headers=headers, json_body=body, provider_label="OpenAI",
)
```

por:

```python
r = _post_with_retry(
    self._endpoint_url(),
    headers=headers,
    json_body=body,
    provider_label=self.provider_label,
)
```

Aplica los mismos cambios a `_translate_dubbing_batch` (también lo mueves al base).

- [ ] **Step 1.2.4: Move `translate_texts` and `translate_for_dubbing` from OpenAITranslator to `_BaseChatTranslator`**

Estos métodos no construyen body HTTP — solo orquestan batches. Cópialos al base SIN modificación. Borra de `OpenAITranslator`.

- [ ] **Step 1.2.5: Reduce `OpenAITranslator` to the 4 abstract methods**

`OpenAITranslator` queda así:

```python
class OpenAITranslator(_BaseChatTranslator):
    """Translate via OpenAI Chat Completions API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        source_lang: str = "EN",
        target_lang: str = "ES",
        temperature: float = 0.2,
        base_url: str | None = None,
    ) -> None:
        super().__init__(source_lang, target_lang)
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not provided (env or constructor)")
        self.api_key = key
        self.model = model
        self.temperature = temperature
        self.provider_label = "OpenAI"
        url = (base_url or OPENAI_URL).rstrip("/")
        if not url.endswith("/chat/completions"):
            if url.endswith("/v1"):
                url = f"{url}/chat/completions"
        self.url = url

    def _endpoint_url(self) -> str:
        return self.url

    def _request_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _wrap_chat_body(self, messages: list[dict], json_mode: bool) -> dict:
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": messages,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    def _extract_message_content(self, resp_json: dict) -> str:
        return resp_json["choices"][0]["message"]["content"]
```

- [ ] **Step 1.2.6: Run snapshot tests — must still pass**

```bash
cd subtitle-generator && python -m pytest tests/test_translator_openai_snapshot.py -v
```

Expected: 3 PASS. Si falla, el refactor rompió algo — revertir y comparar diff antes de re-intentar.

- [ ] **Step 1.2.7: Run the full subtitle-generator test suite**

```bash
cd subtitle-generator && python -m pytest -v -m "not requires_ollama"
```

Expected: todos los tests existentes verdes. Los que ya estaban fallando antes del refactor pueden seguir fallando — no es regresión.

- [ ] **Step 1.2.8: Commit**

```bash
git add subtitle-generator/subtitle_generator/translator.py
git commit -m "refactor(translator): extract _BaseChatTranslator from OpenAITranslator"
```

---

## Phase 2 — `OllamaTranslator` + eliminar DeepL

### Task 2.1: TDD `OllamaTranslator`

**Files:**
- Create: `subtitle-generator/tests/test_translator_ollama.py`
- Modify: `subtitle-generator/subtitle_generator/translator.py`

- [ ] **Step 2.1.1: Write the failing test**

Crea `subtitle-generator/tests/test_translator_ollama.py`:

```python
"""Unit tests for OllamaTranslator (mocked, no requiere Ollama corriendo)."""
from __future__ import annotations

import json
import os
from unittest.mock import patch, MagicMock

import pytest

from subtitle_generator.translator import OllamaTranslator, make_translator


def _mock_ollama_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"message": {"content": content}}
    return resp


def test_default_base_url_and_model():
    t = OllamaTranslator()
    assert t.base_url == "http://ollama:11434"
    assert t.model == "qwen2.5:7b-instruct-q4_K_M"
    assert t._endpoint_url() == "http://ollama:11434/api/chat"
    assert t.provider_label == "Ollama"


def test_base_url_from_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    t = OllamaTranslator()
    assert t.base_url == "http://localhost:11434"


def test_request_headers_no_auth():
    t = OllamaTranslator()
    h = t._request_headers()
    assert h == {"Content-Type": "application/json"}
    assert "Authorization" not in h


def test_wrap_chat_body_json_mode_true():
    t = OllamaTranslator(model="qwen2.5:7b-instruct-q4_K_M", temperature=0.2)
    msgs = [{"role": "user", "content": "hi"}]
    body = t._wrap_chat_body(msgs, json_mode=True)
    assert body == {
        "model": "qwen2.5:7b-instruct-q4_K_M",
        "messages": msgs,
        "stream": False,
        "options": {"temperature": 0.2},
        "format": "json",
    }


def test_wrap_chat_body_json_mode_false_omits_format():
    t = OllamaTranslator()
    body = t._wrap_chat_body([], json_mode=False)
    assert "format" not in body


def test_extract_message_content():
    t = OllamaTranslator()
    assert t._extract_message_content({"message": {"content": "hola"}}) == "hola"


def test_translate_texts_e2e_mocked():
    t = OllamaTranslator()
    fake = _mock_ollama_response('{"t":["hola"]}')
    with patch("subtitle_generator.translator._post_with_retry", return_value=fake) as m:
        out = t.translate_texts(["hello"])
    assert out == ["hola"]
    args, kwargs = m.call_args
    assert args[0] == "http://ollama:11434/api/chat"
    body = kwargs["json_body"]
    assert body["format"] == "json"
    assert body["stream"] is False


def test_count_mismatch_retry():
    t = OllamaTranslator()
    short = _mock_ollama_response('{"t":["one"]}')
    correct = _mock_ollama_response('{"t":["one","two"]}')
    with patch(
        "subtitle_generator.translator._post_with_retry",
        side_effect=[short, correct],
    ) as m:
        out = t.translate_texts(["a", "b"])
    assert out == ["one", "two"]
    assert m.call_count == 2


def test_make_translator_ollama():
    t = make_translator("ollama")
    assert isinstance(t, OllamaTranslator)


def test_make_translator_ollama_with_custom_model():
    t = make_translator("ollama", model="qwen2.5:14b-instruct-q4_K_M")
    assert isinstance(t, OllamaTranslator)
    assert t.model == "qwen2.5:14b-instruct-q4_K_M"
```

- [ ] **Step 2.1.2: Run — verify tests fail because OllamaTranslator doesn't exist**

```bash
cd subtitle-generator && python -m pytest tests/test_translator_ollama.py -v
```

Expected: errores `ImportError: cannot import name 'OllamaTranslator'`.

- [ ] **Step 2.1.3: Implement `OllamaTranslator` in `translator.py`**

Añade DESPUÉS de `OpenAITranslator` y ANTES de `make_translator`:

```python
class OllamaTranslator(_BaseChatTranslator):
    """Translate via Ollama native /api/chat with strict JSON mode."""

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
        self.provider_label = "Ollama"
        self.base_url = (
            base_url or os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
        ).rstrip("/")

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

- [ ] **Step 2.1.4: Update `make_translator` factory: add ollama, remove deepl**

Reemplaza la función `make_translator` entera por:

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
    """Build a translator for the requested provider name."""
    p = (provider or "").lower().strip()
    if p == "ollama":
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

- [ ] **Step 2.1.5: Run the new tests**

```bash
cd subtitle-generator && python -m pytest tests/test_translator_ollama.py -v
```

Expected: 10 PASS.

- [ ] **Step 2.1.6: Commit**

```bash
git add subtitle-generator/subtitle_generator/translator.py subtitle-generator/tests/test_translator_ollama.py
git commit -m "feat(translator): add OllamaTranslator with /api/chat + format=json"
```

---

### Task 2.2: Eliminar `DeepLTranslator` y constantes DeepL

**Files:**
- Modify: `subtitle-generator/subtitle_generator/translator.py`

- [ ] **Step 2.2.1: Remove `DeepLTranslator` class**

En `translator.py`, borra ENTERA la sección `# DeepL provider` y la clase `DeepLTranslator` (líneas ~158-214 del archivo actual).

- [ ] **Step 2.2.2: Remove DeepL URL constants**

Borra las líneas:
```python
DEEPL_FREE_URL = "https://api-free.deepl.com/v2/translate"
DEEPL_PRO_URL = "https://api.deepl.com/v2/translate"
```

- [ ] **Step 2.2.3: Update module docstring**

Reemplaza la primera línea del archivo:
```python
"""SRT translation with pluggable providers (OpenAI, DeepL).
```
por:
```python
"""SRT translation with pluggable providers (Ollama local, OpenAI cloud).
```

- [ ] **Step 2.2.4: Run tests — must still pass (no DeepL test was passing)**

```bash
cd subtitle-generator && python -m pytest -v -m "not requires_ollama"
```

Expected: todos verdes excepto cualquier test pre-existente que importe `DeepLTranslator` (los borraremos en Phase 9).

- [ ] **Step 2.2.5: Commit**

```bash
git add subtitle-generator/subtitle_generator/translator.py
git commit -m "refactor(translator): remove DeepLTranslator and DeepL constants"
```

---

## Phase 3 — Refactor `subtitle-generator/app.py`

### Task 3.1: Eliminar ramas DeepL en `_build_translator_with_fallback`

**Files:**
- Modify: `subtitle-generator/app.py`

- [ ] **Step 3.1.1: Replace the function**

Reemplaza `_build_translator_with_fallback` (líneas ~163-209) por:

```python
def _build_translator_with_fallback(opts: dict):
    """Build (primary, fallback) translators from job options.

    Primary provider defaults to ``ollama``. Fallback is used only if the
    primary raises at translate time. Ollama no necesita api_key.
    """
    from subtitle_generator.translator import make_translator  # type: ignore

    provider = (opts.get("provider") or "ollama").lower()
    primary_key = opts.get("api_key")
    if not primary_key and provider == "openai":
        primary_key = os.environ.get("OPENAI_API_KEY")

    primary = make_translator(
        provider,
        api_key=primary_key,
        source_lang=opts.get("source_lang", "EN"),
        target_lang=opts.get("target_lang", "ES"),
        model=opts.get("model"),
        formality=opts.get("formality"),
    )

    fb_name = (opts.get("fallback_provider") or "").lower() or None
    fb = None
    if fb_name and fb_name != provider:
        fb_key = opts.get("fallback_api_key")
        if not fb_key and fb_name == "openai":
            fb_key = os.environ.get("OPENAI_API_KEY")
        try:
            fb = make_translator(
                fb_name,
                api_key=fb_key,
                source_lang=opts.get("source_lang", "EN"),
                target_lang=opts.get("target_lang", "ES"),
                model=opts.get("fallback_model"),
                formality=opts.get("formality"),
            )
        except ValueError:
            fb = None
    return primary, fb
```

- [ ] **Step 3.1.2: Update `_translate_for_dubbing` isinstance check**

En `subtitle-generator/app.py` línea 270, busca:
```python
from subtitle_generator.translator import OpenAITranslator  # type: ignore

if not isinstance(primary, OpenAITranslator):
    raise RuntimeError(
        "dubbing_mode requires OpenAI provider; DeepL has no budget-aware mode"
    )
```

Reemplaza por:
```python
from subtitle_generator.translator import _BaseChatTranslator  # type: ignore

if not isinstance(primary, _BaseChatTranslator):
    raise RuntimeError(
        "dubbing_mode requires a chat-based translator (Ollama or OpenAI)"
    )
```

- [ ] **Step 3.1.3: Update docstring of `_run_subtitle_generator`**

Línea ~95: cambia `EN→ES via DeepL` por `EN→ES via Ollama/OpenAI`.

- [ ] **Step 3.1.4: Verify no remaining DeepL references**

```bash
grep -n -i "deepl" subtitle-generator/app.py
```

Expected: sin output.

- [ ] **Step 3.1.5: Run tests**

```bash
cd subtitle-generator && python -m pytest -v -m "not requires_ollama"
```

Expected: verdes (excepto los pre-existentes que ya fallaban).

- [ ] **Step 3.1.6: Commit**

```bash
git add subtitle-generator/app.py
git commit -m "refactor(subtitle-generator): default provider=ollama, drop DeepL branches"
```

---

### Task 3.2: Mejorar fallback dubbing-aware

**Files:**
- Modify: `subtitle-generator/app.py`

Cuando el primary es `_BaseChatTranslator` (Ollama o OpenAI) y falla en dubbing_mode, el fallback debería intentar también modo dubbing antes de caer a literal.

- [ ] **Step 3.2.1: Patch `_run_translate_directory` dubbing branch**

En `subtitle-generator/app.py`, busca el bloque que empieza con `elif dubbing_mode:` (línea ~428). Reemplaza el `except Exception` posterior:

```python
except Exception as exc:
    emit(JobEvent(type="log", data={"message":
        f"{provider_name} dub-compact failed on {srt.name}: {exc}"
    }))
    if fallback is not None:
        try:
            fallback.translate_srt(srt, sub_out)
            emit(JobEvent(type="log", data={"message":
                f"translated subs via fallback {fb_name} (literal): {srt.name}"
            }))
        except Exception as exc2:
            emit(JobEvent(type="log", data={"message":
                f"ERROR fallback {fb_name} also failed on {srt.name}: {exc2}"
            }))
```

por:

```python
except Exception as exc:
    emit(JobEvent(type="log", data={"message":
        f"{provider_name} dub-compact failed on {srt.name}: {exc}"
    }))
    if fallback is not None:
        from subtitle_generator.translator import _BaseChatTranslator  # type: ignore
        try:
            if isinstance(fallback, _BaseChatTranslator):
                _translate_for_dubbing(fallback, srt, sub_out, cps, force_slot_mode=False)
                emit(JobEvent(type="log", data={"message":
                    f"translated subs via fallback {fb_name} (dub-aware): {srt.name}"
                }))
            else:
                fallback.translate_srt(srt, sub_out)
                emit(JobEvent(type="log", data={"message":
                    f"translated subs via fallback {fb_name} (literal): {srt.name}"
                }))
        except Exception as exc2:
            emit(JobEvent(type="log", data={"message":
                f"ERROR fallback {fb_name} also failed on {srt.name}: {exc2}"
            }))
```

- [ ] **Step 3.2.2: Run tests**

```bash
cd subtitle-generator && python -m pytest -v -m "not requires_ollama"
```

Expected: verdes.

- [ ] **Step 3.2.3: Commit**

```bash
git add subtitle-generator/app.py
git commit -m "feat(subtitle-generator): dubbing-aware fallback when both providers chat-based"
```

---

## Phase 4 — Infra Ollama

### Task 4.1: Crear entrypoint Ollama

**Files:**
- Create: `ollama/entrypoint.sh`

- [ ] **Step 4.1.1: Create the entrypoint script**

```bash
mkdir -p ollama
```

Escribe `ollama/entrypoint.sh`:

```bash
#!/bin/bash
set -e

# Arrancar ollama en background
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

wait $OLLAMA_PID
```

- [ ] **Step 4.1.2: Make it executable**

```bash
chmod +x ollama/entrypoint.sh
```

- [ ] **Step 4.1.3: Commit**

```bash
git add ollama/entrypoint.sh
git commit -m "feat(infra): add ollama entrypoint with idempotent model pull"
```

---

### Task 4.2: Añadir servicio `ollama` al docker-compose.yml

**Files:**
- Modify: `docker-compose.yml`

- [ ] **Step 4.2.1: Inspect current compose structure**

```bash
grep -n "^services:\|^volumes:\|^networks:\|bjj_net" docker-compose.yml | head -20
```

Toma nota de los nombres exactos de la red bridge y la sección `volumes:` top-level.

- [ ] **Step 4.2.2: Add `ollama` service**

Añade el siguiente bloque dentro de `services:` (después de `dubbing-generator` o donde encaje en el orden alfabético):

```yaml
  ollama:
    image: ollama/ollama:latest
    container_name: bjj-ollama
    restart: unless-stopped
    networks:
      - bjj_net
    ports:
      - "11434:11434"
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
      start_period: 1200s
```

Si la red bridge tiene otro nombre (ej. `default` o `bjj-network`), ajusta `networks:` para que coincida con el resto de servicios.

- [ ] **Step 4.2.3: Add `ollama-models` volume**

En la sección `volumes:` top-level, añade:

```yaml
  ollama-models:
```

- [ ] **Step 4.2.4: Validate compose syntax**

```bash
docker compose config --quiet
```

Expected: sin errores. Si sale "service 'ollama' refers to undefined network bjj_net", corregir el nombre de la red.

- [ ] **Step 4.2.5: Commit**

```bash
git add docker-compose.yml
git commit -m "feat(infra): add ollama service with GPU passthrough and pull-on-start"
```

---

## Phase 5 — Backend API (paralelizable: 5.1 / 5.2 / 5.3 son disjuntos)

### Task 5.1: `processor-api/api/settings.py` — defaults + auto-migración

**Files:**
- Modify: `processor-api/api/settings.py`

- [ ] **Step 5.1.1: Update `_DEFAULTS`**

En `processor-api/api/settings.py` línea ~31, cambia:

```python
"deepl_api_key": None,
"openai_api_key": None,
"translation_provider": "openai",
"translation_model": "gpt-4o",
"translation_fallback_provider": "deepl",
```

por:

```python
"openai_api_key": None,
"translation_provider": "ollama",
"translation_model": "qwen2.5:7b-instruct-q4_K_M",
"translation_fallback_provider": "openai",
```

(Se elimina `"deepl_api_key"`.)

- [ ] **Step 5.1.2: Add `_migrate_legacy_translation_settings` helper**

Añade DESPUÉS de `_maybe_import_legacy_json` (línea ~111):

```python
def _migrate_legacy_translation_settings() -> None:
    """One-shot rewrite de valores legacy (deepl → ollama). Idempotente.

    Solo migra si el usuario tenía explícitamente DeepL configurado. Respeta
    a quien tenga ``translation_provider="openai"`` (no fuerza Ollama).
    """
    try:
        with session_scope() as s:
            row = s.get(Setting, "translation_provider")
            if row is not None and json.loads(row.value) == "deepl":
                row.value = json.dumps("ollama")
                model_row = s.get(Setting, "translation_model")
                if model_row is not None:
                    model_row.value = json.dumps("qwen2.5:7b-instruct-q4_K_M")

            fb_row = s.get(Setting, "translation_fallback_provider")
            if fb_row is not None and json.loads(fb_row.value) == "deepl":
                fb_row.value = json.dumps("openai")

            deepl_row = s.get(Setting, "deepl_api_key")
            if deepl_row is not None:
                s.delete(deepl_row)
    except Exception as exc:
        log.warning("legacy translation settings migration failed: %s", exc)
```

- [ ] **Step 5.1.3: Wire migration into `_ensure_initialized`**

En `_ensure_initialized` (línea ~80), añade tras `_maybe_import_legacy_json()`:

```python
_migrate_legacy_translation_settings()
```

Queda así:

```python
def _ensure_initialized() -> None:
    global _initialized
    if _initialized:
        return
    try:
        init_db()
    except Exception as exc:
        log.warning("init_db failed: %s", exc)
    _maybe_import_legacy_json()
    _migrate_legacy_translation_settings()
    _initialized = True
```

- [ ] **Step 5.1.4: Update PUT handler — remove `deepl_api_key`, restrict provider enum**

Busca en `settings.py` el handler PUT (líneas ~200-260). Borra el bloque:

```python
if "deepl_api_key" in body:
    dk = body["deepl_api_key"]
    if dk is not None and not isinstance(dk, str):
        return JSONResponse({"error": "deepl_api_key must be a string or null"}, status_code=422)
    current["deepl_api_key"] = dk.strip() if isinstance(dk, str) else dk
```

Y donde valide `translation_provider` / `translation_fallback_provider`, asegúrate que solo acepta `("ollama", "openai")` para provider y `("", "ollama", "openai")` para fallback. Si el código actual tiene un loop genérico `for k in ("translation_provider", ...)`, añade explícitamente:

```python
if "translation_provider" in body:
    val = body["translation_provider"]
    if val not in ("ollama", "openai"):
        return JSONResponse(
            {"error": "translation_provider must be 'ollama' or 'openai'"},
            status_code=422,
        )
    current["translation_provider"] = val

if "translation_fallback_provider" in body:
    val = body["translation_fallback_provider"]
    if val not in ("", "ollama", "openai", None):
        return JSONResponse(
            {"error": "translation_fallback_provider must be '', 'ollama', 'openai' or null"},
            status_code=422,
        )
    current["translation_fallback_provider"] = val if val else None
```

(Si el handler ya tenía un loop genérico que escribía `current[k] = body[k]`, elimina del loop esas dos claves para evitar doble-escritura.)

- [ ] **Step 5.1.5: Run processor-api tests**

```bash
cd processor-api && python -m pytest tests/ -v -k "settings or translation"
```

Expected: si los tests existentes referencian `deepl_api_key`, fallarán — eso lo arreglamos en Phase 9. Si fallan otros, revisar.

- [ ] **Step 5.1.6: Commit**

```bash
git add processor-api/api/settings.py
git commit -m "feat(api/settings): default ollama, drop deepl, auto-migrate legacy values"
```

---

### Task 5.2: `processor-api/api/pipeline.py` — defaults + endpoint flush-ollama

**Files:**
- Modify: `processor-api/api/pipeline.py`

- [ ] **Step 5.2.1: Replace deepl branches in pipeline.py**

En `processor-api/api/pipeline.py` línea ~369, busca el bloque que mapea provider → api_key:

```python
key=(
    get_setting("openai_api_key") if provider == "openai"
    else get_setting("deepl_api_key") if provider == "deepl"
    else None
)
```

Reemplaza por:

```python
key=(
    get_setting("openai_api_key") if provider == "openai"
    else None  # ollama no necesita key
)
```

Repite el mismo cambio en la rama de fallback (línea ~415).

- [ ] **Step 5.2.2: Update default provider**

Busca:
```python
provider = (options.get("provider") or get_setting("translation_provider") or "openai").lower()
```

Cambia `"openai"` por `"ollama"`.

- [ ] **Step 5.2.3: Add `/api/pipeline/flush-ollama` endpoint**

Busca el endpoint `flush-gpu` existente (debería estar en pipeline.py). Tras él, añade:

```python
@router.post("/flush-ollama")
async def flush_ollama() -> dict:
    """Descarga el modelo de Ollama de VRAM al instante.

    Útil entre fases del pipeline secuencial: tras translate, antes de Kokoro,
    para liberar VRAM (~4.5 GB con qwen2.5:7b-Q4) y evitar OOM.
    """
    import httpx

    model = get_setting("translation_model") or "qwen2.5:7b-instruct-q4_K_M"
    base = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                f"{base}/api/chat",
                json={"model": model, "messages": [], "keep_alive": 0, "stream": False},
            )
        return {"ok": r.status_code < 400, "status": r.status_code}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
```

Si `os` no está importado en `pipeline.py`, añade `import os` arriba.

- [ ] **Step 5.2.4: Verify no DeepL references remain**

```bash
grep -n -i "deepl" processor-api/api/pipeline.py
```

Expected: sin output.

- [ ] **Step 5.2.5: Run tests**

```bash
cd processor-api && python -m pytest tests/ -v -k "pipeline"
```

Expected: verdes (excepto tests pre-existentes).

- [ ] **Step 5.2.6: Commit**

```bash
git add processor-api/api/pipeline.py
git commit -m "feat(api/pipeline): default ollama provider, add /flush-ollama endpoint"
```

---

### Task 5.3: `processor-api/api/subtitles.py` — rama ollama en `_key_for_provider`

**Files:**
- Modify: `processor-api/api/subtitles.py`

- [ ] **Step 5.3.1: Read current `_key_for_provider` and surrounding code**

Lee `processor-api/api/subtitles.py` líneas 130-170 para localizar:
- La función `_key_for_provider` (líneas ~132-145).
- El default `provider = "openai"` (línea ~144).
- El check `raise HTTPException(... API key missing ...)` (línea ~165).

- [ ] **Step 5.3.2: Replace `_key_for_provider`**

Reemplaza la función entera por:

```python
def _key_for_provider(provider: str) -> Optional[str]:
    """Resuelve la api_key del provider. Ollama no la necesita."""
    p = (provider or "").lower().strip()
    if p == "openai":
        return get_setting("openai_api_key")
    if p == "ollama":
        return None
    return None
```

- [ ] **Step 5.3.3: Update default provider**

Busca:
```python
provider = (req.provider or "openai").lower()
```
o similar (línea ~144). Cambia `"openai"` por `"ollama"`.

- [ ] **Step 5.3.4: Update api-key-missing check**

Busca el check:
```python
if not key:
    raise HTTPException(...)
```

Reemplaza por:
```python
if provider != "ollama" and not key:
    raise HTTPException(
        status_code=400,
        detail=f"{provider} API key missing in settings",
    )
```

- [ ] **Step 5.3.5: Verify no DeepL refs**

```bash
grep -n -i "deepl" processor-api/api/subtitles.py
```

Expected: sin output.

- [ ] **Step 5.3.6: Run tests**

```bash
cd processor-api && python -m pytest tests/ -v -k "subtitles"
```

Expected: verdes (los tests pre-existentes que esperan provider=openai por defecto pueden fallar — adaptarlos en step 9).

- [ ] **Step 5.3.7: Commit**

```bash
git add processor-api/api/subtitles.py
git commit -m "feat(api/subtitles): default ollama, ollama branch in _key_for_provider"
```

---

## Phase 6 — Wire central (NO paralelizar — toca app.py central)

### Task 6.1: Limpiar deepl en `processor-api/api/app.py` y extender `health_proxy.py`

**Files:**
- Modify: `processor-api/api/app.py`
- Modify: `processor-api/api/health_proxy.py`

- [ ] **Step 6.1.1: Locate any deepl references in app.py**

```bash
grep -n -i "deepl\|DEEPL" processor-api/api/app.py
```

Si hay matches, eliminarlos (probablemente proxy de api-key o branch en `run_translation`). Si no hay matches, este sub-step queda no-op.

- [ ] **Step 6.1.2: Extend `BACKENDS` dict in health_proxy.py**

En `processor-api/api/health_proxy.py` línea 14, añade:

```python
BACKENDS: dict[str, str] = {
    "chapter-splitter": os.environ.get("SPLITTER_URL", "http://chapter-splitter:8001"),
    "subtitle-generator": os.environ.get("SUBS_URL", "http://subtitle-generator:8002"),
    "dubbing-generator": os.environ.get("DUBBING_URL", "http://dubbing-generator:8003"),
    "ollama": os.environ.get("OLLAMA_URL", "http://ollama:11434"),
}
```

- [ ] **Step 6.1.3: Override `_ping` for ollama (uses /api/tags, not /health)**

El `_ping` actual hace `GET {base}/health`. Ollama no expone `/health` — usa `/api/tags`. Modifica `_ping`:

```python
async def _ping(client: httpx.AsyncClient, service: str, base: str) -> dict:
    health_path = "/api/tags" if service == "ollama" else "/health"
    try:
        r = await client.get(f"{base}{health_path}", timeout=3.0)
        if r.status_code == 200:
            body = r.json()
            # Para ollama, el body es {"models": [...]} — devolverlo crudo
            return {"service": service, "status": "up", "body": body}
        return {"service": service, "status": "down", "error": f"HTTP {r.status_code}"}
    except Exception as exc:
        return {"service": service, "status": "down", "error": str(exc)}
```

- [ ] **Step 6.1.4: Run tests for health_proxy**

```bash
cd processor-api && python -m pytest tests/ -v -k "health"
```

Expected: verdes (los tests existentes esperan los 3 servicios; si miden el array length, hay que actualizarlos en 9).

- [ ] **Step 6.1.5: Commit**

```bash
git add processor-api/api/app.py processor-api/api/health_proxy.py
git commit -m "feat(api/health): expose ollama in /health/backends aggregate

WIRE_TRANSLATOR_OLLAMA: api/app.py deepl branches removed; api/health_proxy.py /health/backends extended with ollama entry"
```

---

## Phase 7 — Frontend

### Task 7.1: Schema, defaults, selectors core (8a)

**Files:**
- Modify: `processor-frontend/src/features/settings/pages/SettingsPage.jsx`

- [ ] **Step 7.1.1: Update Zod schema**

Busca en `SettingsPage.jsx` líneas ~125-135 el schema. Encuentra:
```js
translation_provider: z.enum(['openai', 'deepl']).optional(),
translation_fallback_provider: z.enum(['', 'openai', 'deepl']).optional(),
deepl_api_key: z.string().optional(),
```

Reemplaza por:
```js
translation_provider: z.enum(['ollama', 'openai']),
translation_fallback_provider: z.enum(['', 'ollama', 'openai']).optional(),
```

(`deepl_api_key` se elimina del schema.)

- [ ] **Step 7.1.2: Update defaultValues**

En el `useForm({...})`, busca `defaultValues:` y cambia:
```js
translation_provider: 'openai',
translation_model: 'gpt-4o-mini',
translation_fallback_provider: 'deepl',
deepl_api_key: '',
```

por:
```js
translation_provider: 'ollama',
translation_model: 'qwen2.5:7b-instruct-q4_K_M',
translation_fallback_provider: 'openai',
```

- [ ] **Step 7.1.3: Update CardDescription literal**

Busca `OpenAI (gpt-4o-mini) por defecto, DeepL como fallback` (línea ~1019). Reemplaza por:

```jsx
Ollama local (qwen2.5-7b) por defecto, OpenAI como fallback opt-in.
```

- [ ] **Step 7.1.4: Update provider Select options**

Busca el `<Select>` de provider. Cambia las opciones:
```jsx
<SelectItem value="openai">OpenAI</SelectItem>
<SelectItem value="deepl">DeepL</SelectItem>
```
por:
```jsx
<SelectItem value="ollama">Ollama (local, gratis)</SelectItem>
<SelectItem value="openai">OpenAI (cloud, pago)</SelectItem>
```

- [ ] **Step 7.1.5: Update fallback provider Select**

Idem para el selector de fallback. Las opciones quedan:
```jsx
<SelectItem value="">Ninguno</SelectItem>
<SelectItem value="ollama">Ollama (local)</SelectItem>
<SelectItem value="openai">OpenAI (cloud)</SelectItem>
```

- [ ] **Step 7.1.6: Conditional model dropdown**

Busca el dropdown de modelo (líneas ~1056-1077, hardcodeado a opciones OpenAI). Reemplaza el bloque por:

```jsx
{provider === 'ollama' ? (
  <Select
    value={watch('translation_model') || 'qwen2.5:7b-instruct-q4_K_M'}
    onValueChange={(v) => setValue('translation_model', v, { shouldDirty: true })}
  >
    <SelectTrigger><SelectValue /></SelectTrigger>
    <SelectContent>
      <SelectItem value="qwen2.5:7b-instruct-q4_K_M">qwen2.5:7b-instruct-q4_K_M (default, ~4.5 GB VRAM)</SelectItem>
      <SelectItem value="qwen2.5:14b-instruct-q4_K_M">qwen2.5:14b-instruct-q4_K_M (~9 GB VRAM, mejor calidad)</SelectItem>
    </SelectContent>
  </Select>
) : (
  <Select
    value={watch('translation_model') || 'gpt-4o-mini'}
    onValueChange={(v) => setValue('translation_model', v, { shouldDirty: true })}
  >
    <SelectTrigger><SelectValue /></SelectTrigger>
    <SelectContent>
      <SelectItem value="gpt-4o-mini">gpt-4o-mini</SelectItem>
      <SelectItem value="gpt-4o">gpt-4o</SelectItem>
      <SelectItem value="gpt-4.1-mini">gpt-4.1-mini</SelectItem>
      <SelectItem value="gpt-4.1">gpt-4.1</SelectItem>
    </SelectContent>
  </Select>
)}
```

(Asume `provider = watch('translation_provider')` ya existe arriba; si no, añadirlo.)

- [ ] **Step 7.1.7: Reset model when provider changes**

Justo después de declarar `provider = watch(...)`, añade:

```jsx
const OLLAMA_MODELS = ['qwen2.5:7b-instruct-q4_K_M', 'qwen2.5:14b-instruct-q4_K_M']
const OPENAI_MODELS = ['gpt-4o-mini', 'gpt-4o', 'gpt-4.1-mini', 'gpt-4.1']

useEffect(() => {
  const current = watch('translation_model')
  if (provider === 'ollama' && !OLLAMA_MODELS.includes(current)) {
    setValue('translation_model', 'qwen2.5:7b-instruct-q4_K_M', { shouldDirty: true })
  } else if (provider === 'openai' && !OPENAI_MODELS.includes(current)) {
    setValue('translation_model', 'gpt-4o-mini', { shouldDirty: true })
  }
}, [provider, setValue, watch])
```

(Si `useEffect` no está importado: `import { useEffect } from 'react'`.)

- [ ] **Step 7.1.8: Remove all DeepL JSX**

Busca y elimina:
- El bloque `<input>` de `deepl_api_key` (label, input, link a deepl.com, badge Free/Pro).
- Estado `showDeepL`, `hasDeepL`, `deeplIsFree`, `deeplBadge`.
- Cualquier `useEffect` que toque `deepl_api_key`.

```bash
grep -n -i "deepl" processor-frontend/src/features/settings/pages/SettingsPage.jsx
```

Expected tras cleanup: sin output.

- [ ] **Step 7.1.9: Build the frontend**

```bash
cd processor-frontend && npm run build
```

Expected: build exitoso. Si Zod / react-hook-form se quejan de que un campo no está en el schema, revisar 7.1.1.

- [ ] **Step 7.1.10: Commit**

```bash
git add processor-frontend/src/features/settings/pages/SettingsPage.jsx
git commit -m "feat(frontend/settings): switch translation provider to ollama+openai, drop deepl UI"
```

---

### Task 7.2: Banner Ollama-down + health pill (8b)

**Files:**
- Modify: `processor-frontend/src/components/layout/useBackendsHealth.js`
- Modify: `processor-frontend/src/features/settings/pages/SettingsPage.jsx`

- [ ] **Step 7.2.1: Add ollama to BACKENDS array**

En `useBackendsHealth.js` línea 4, cambia:

```js
export const BACKENDS = [
  { id: 'chapter-splitter', label: 'Chapter Splitter' },
  { id: 'subtitle-generator', label: 'Subtitle Generator' },
  { id: 'dubbing-generator', label: 'Dubbing Generator' },
]
```

por:

```js
export const BACKENDS = [
  { id: 'chapter-splitter', label: 'Chapter Splitter' },
  { id: 'subtitle-generator', label: 'Subtitle Generator' },
  { id: 'dubbing-generator', label: 'Dubbing Generator' },
  { id: 'ollama', label: 'Ollama' },
]
```

- [ ] **Step 7.2.2: Add Ollama-down banner in SettingsPage**

Importa `useBackendsHealth` arriba de `SettingsPage.jsx`:

```jsx
import { useBackendsHealth } from '@/components/layout/useBackendsHealth'
```

Dentro del componente, tras los `watch(...)`:

```jsx
const backends = useBackendsHealth()
const ollamaStatus = backends.find((b) => b.id === 'ollama')?.status
const showOllamaWarning = provider === 'ollama' && ollamaStatus === 'down'
```

En el JSX, justo encima del Card de Translation, añade:

```jsx
{showOllamaWarning && (
  <div className="rounded-md border border-yellow-500/50 bg-yellow-500/10 p-3 text-sm">
    <p className="font-medium">Ollama no está respondiendo</p>
    <p className="mt-1 text-muted-foreground">
      El modelo se está descargando (~4.5 GB la primera vez) o el servicio no arrancó.
      Mientras tanto puedes cambiar a OpenAI en el selector.
    </p>
  </div>
)}
```

- [ ] **Step 7.2.3: Build**

```bash
cd processor-frontend && npm run build
```

Expected: build exitoso.

- [ ] **Step 7.2.4: Commit**

```bash
git add processor-frontend/src/components/layout/useBackendsHealth.js processor-frontend/src/features/settings/pages/SettingsPage.jsx
git commit -m "feat(frontend): add ollama health pill + down banner in settings"
```

---

## Phase 8 — Cleanup código muerto

### Task 8.1: Borrar dubbing-generator/translation/

**Files:**
- Delete: `dubbing-generator/dubbing_generator/translation/translator.py`
- Delete: `dubbing-generator/dubbing_generator/translation/__init__.py` (si solo importaba el translator)
- Delete: `dubbing-generator/tests/test_translator.py`

- [ ] **Step 8.1.1: Verify no remaining imports**

```bash
grep -rn "from dubbing_generator.translation" dubbing-generator/ --include="*.py"
```

Expected: solo el propio archivo y su test.

- [ ] **Step 8.1.2: Inspect __init__.py before deleting**

```bash
cat dubbing-generator/dubbing_generator/translation/__init__.py
```

Si contiene solo el re-export del translator, borrar. Si contiene otra cosa, conservar.

- [ ] **Step 8.1.3: Delete files**

```bash
rm dubbing-generator/dubbing_generator/translation/translator.py
rm dubbing-generator/tests/test_translator.py
# Solo si __init__.py era trivial:
rm dubbing-generator/dubbing_generator/translation/__init__.py
rmdir dubbing-generator/dubbing_generator/translation/ 2>/dev/null || true
```

- [ ] **Step 8.1.4: Run dubbing-generator tests**

```bash
cd dubbing-generator && python -m pytest -v
```

Expected: verdes (o al menos sin nuevas roturas atribuibles a este step).

- [ ] **Step 8.1.5: Commit**

```bash
git add -A dubbing-generator/
git commit -m "chore(dubbing-generator): remove dead Helsinki-NLP translator"
```

---

### Task 8.2: Eliminar `DEEPL_API_KEY` de .env.example

**Files:**
- Modify: `.env.example`

- [ ] **Step 8.2.1: Remove DEEPL line**

```bash
grep -n "DEEPL" .env.example
```

Si hay match, edita el archivo y elimina la línea `DEEPL_API_KEY=...`.

- [ ] **Step 8.2.2: Commit**

```bash
git add .env.example
git commit -m "chore: drop DEEPL_API_KEY from .env.example"
```

---

### Task 8.3: Adaptar tests pre-existentes que referencian DeepL

**Files:**
- Modify (o delete): `subtitle-generator/tests/test_translator.py` (si referencia DeepLTranslator)
- Modify: `processor-api/tests/test_settings.py` (si valida deepl_api_key)
- Modify: cualquier otro test que mencione deepl

- [ ] **Step 8.3.1: Find all deepl test references**

```bash
grep -rn -i "deepl" subtitle-generator/tests/ processor-api/tests/ processor-frontend/tests/ 2>/dev/null
```

- [ ] **Step 8.3.2: For each match, decide: delete or adapt**

- Tests que importan `DeepLTranslator` o `DeepLConfig`: **borrar el test** (clase ya no existe).
- Tests que validan `deepl_api_key` en settings: **borrar el bloque** correspondiente.
- Tests que esperan `provider="openai"` por defecto: **adaptar a `provider="ollama"`**.

- [ ] **Step 8.3.3: Run all suites**

```bash
cd subtitle-generator && python -m pytest -v -m "not requires_ollama"
cd ../processor-api && python -m pytest -v
```

Expected: todos verdes.

- [ ] **Step 8.3.4: Commit**

```bash
git add subtitle-generator/tests/ processor-api/tests/
git commit -m "test: drop DeepL-specific tests, update default provider in expectations"
```

---

## Phase 9 — Test de calidad (golden set)

### Task 9.1: Registrar pytest markers

**Files:**
- Modify: `subtitle-generator/pyproject.toml`

- [ ] **Step 9.1.1: Add markers section**

Edita `subtitle-generator/pyproject.toml`. En la sección `[tool.pytest.ini_options]`, añade:

```toml
markers = [
    "requires_ollama: requiere servicio Ollama corriendo en localhost:11434",
    "quality: test de calidad sobre golden set BJJ",
]
```

Si la sección no existe aún, créala completa.

- [ ] **Step 9.1.2: Verify pytest recognizes them**

```bash
cd subtitle-generator && python -m pytest --markers | grep -E "requires_ollama|quality"
```

Expected: 2 líneas con las descripciones.

- [ ] **Step 9.1.3: Commit**

```bash
git add subtitle-generator/pyproject.toml
git commit -m "test(subtitle-generator): register pytest markers requires_ollama+quality"
```

---

### Task 9.2: Crear golden set BJJ

**Files:**
- Create: `subtitle-generator/tests/data/bjj_golden_set.jsonl`

- [ ] **Step 9.2.1: Create directory**

```bash
mkdir -p subtitle-generator/tests/data
```

- [ ] **Step 9.2.2: Write the golden set**

Crea `subtitle-generator/tests/data/bjj_golden_set.jsonl` (un JSON object por línea):

```jsonl
{"en": "Get your grip on the sleeve and pass the guard.", "must_keep_en": ["grip", "guard"], "duration_ms": 2200}
{"en": "Use the butterfly hook to sweep him.", "must_keep_en": ["butterfly hook", "sweep"], "duration_ms": 1900}
{"en": "Establish your underhook and frame on the hip.", "must_keep_en": ["underhook", "frame"], "duration_ms": 2300}
{"en": "From half guard, threaten the kimura to force the pass.", "must_keep_en": ["half guard", "kimura"], "duration_ms": 2700}
{"en": "Control the ankle, then break the grip on your collar.", "must_keep_en": ["grip", "collar"], "duration_ms": 2400}
{"en": "Pummel through to get the underhook and crossface.", "must_keep_en": ["underhook", "crossface"], "duration_ms": 2500}
{"en": "Set up the heel hook from ashi garami.", "must_keep_en": ["heel hook", "ashi garami"], "duration_ms": 2200}
{"en": "Backstep into the leg drag and finish the pass.", "must_keep_en": ["leg drag", "pass"], "duration_ms": 2400}
{"en": "Frame against the cross face to recover guard.", "must_keep_en": ["frame", "cross face", "guard"], "duration_ms": 2600}
{"en": "From de la Riva, hook the leg and start the berimbolo.", "must_keep_en": ["de la Riva", "hook", "berimbolo"], "duration_ms": 2900}
{"en": "Drill the entry to x-guard from butterfly hook.", "must_keep_en": ["x-guard", "butterfly hook"], "duration_ms": 2500}
{"en": "Stack pass when his knees are on your chest.", "must_keep_en": ["stack pass"], "duration_ms": 2100}
{"en": "Sit through to north-south kimura.", "must_keep_en": ["north-south", "kimura"], "duration_ms": 1900}
{"en": "From the back, attack with bow and arrow choke.", "must_keep_en": ["bow and arrow"], "duration_ms": 2300}
{"en": "Post on his hip and shrimp out to recover.", "must_keep_en": ["post", "shrimp"], "duration_ms": 2300}
{"en": "Threaten the omoplata to force him to roll.", "must_keep_en": ["omoplata"], "duration_ms": 2200}
{"en": "Use the lapel grip to set up the worm guard.", "must_keep_en": ["lapel", "grip", "worm guard"], "duration_ms": 2700}
{"en": "Hand fight to break the underhook battle.", "must_keep_en": ["underhook"], "duration_ms": 2100}
{"en": "Knee cut pass straight into mount.", "must_keep_en": ["knee cut", "mount"], "duration_ms": 2000}
{"en": "Reset and try the toreando from standing.", "must_keep_en": ["toreando"], "duration_ms": 2100}
```

- [ ] **Step 9.2.3: Commit**

```bash
git add subtitle-generator/tests/data/bjj_golden_set.jsonl
git commit -m "test(quality): add BJJ golden set for translation quality tests"
```

---

### Task 9.3: Test calidad automatizable

**Files:**
- Create: `subtitle-generator/tests/test_translator_quality.py`

- [ ] **Step 9.3.1: Write the quality test**

Crea `subtitle-generator/tests/test_translator_quality.py`:

```python
"""Quality tests on BJJ golden set.

Requires Ollama running at localhost:11434 with qwen2.5:7b-instruct-q4_K_M
pulled. Skipped by default — run with `pytest -m requires_ollama`.

Métricas:
- Glosario: % de ítems donde TODOS los must_keep_en aparecen literales en ES.
- Peninsular: % SIN latinismos prohibidos Y CON ≥1 marcador peninsular.
- Char budget: % en rango [80%, 110%] del budget cuando fill_budget=True.

Pasa: las 3 ≥ 85%. Falla si <70%. Warning entre 70-85%.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from subtitle_generator.translator import OllamaTranslator


GOLDEN_PATH = Path(__file__).parent / "data" / "bjj_golden_set.jsonl"

LATAM_RE = re.compile(
    r"\b(ahorita|ustedes|agarrar|agarra|agarro|agarras|chévere|chido|párate|parate|nomás|porfa)\b",
    re.IGNORECASE,
)
PENINSULAR_RE = re.compile(
    r"\b(coge|coges|fíjate|vale|tú|tus|hostia|joder|venga|aquí|ahora|mira)\b",
    re.IGNORECASE,
)


def _load_golden() -> list[dict]:
    items = []
    for line in GOLDEN_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(json.loads(line))
    return items


@pytest.mark.requires_ollama
@pytest.mark.quality
def test_translation_quality_on_bjj_golden_set():
    golden = _load_golden()
    assert len(golden) >= 20, "golden set debe tener al menos 20 ítems"

    # Apunta al puerto publicado por docker compose (host → contenedor).
    t = OllamaTranslator(base_url="http://localhost:11434")
    items = [{"text": g["en"], "duration_ms": g["duration_ms"]} for g in golden]
    translated = t.translate_for_dubbing(items, cps=17.0, fill_budget=True)

    assert len(translated) == len(golden), "count mismatch"

    glossary_pass = 0
    peninsular_pass = 0
    budget_pass = 0
    failures = []

    for g, es in zip(golden, translated):
        es_lower = es.lower()

        # (a) Glosario: todos los must_keep_en presentes (case-insensitive)
        gloss_ok = all(term.lower() in es_lower for term in g["must_keep_en"])
        if gloss_ok:
            glossary_pass += 1
        else:
            missing = [t for t in g["must_keep_en"] if t.lower() not in es_lower]
            failures.append(f"GLOSS  | {g['en'][:60]}... | missing {missing} | got: {es!r}")

        # (b) Peninsular: SIN latam Y CON ≥1 marker
        latam_match = LATAM_RE.search(es)
        peninsular_match = PENINSULAR_RE.search(es)
        pen_ok = (latam_match is None) and (peninsular_match is not None)
        if pen_ok:
            peninsular_pass += 1
        elif latam_match:
            failures.append(f"LATAM  | {es!r} | latinismo: {latam_match.group(0)}")

        # (c) Char budget: en [80%, 110%] del target
        target = int((g["duration_ms"] / 1000.0) * 17.0)
        ratio = len(es) / max(1, target)
        if 0.80 <= ratio <= 1.10:
            budget_pass += 1

    n = len(golden)
    glossary_pct = 100 * glossary_pass / n
    peninsular_pct = 100 * peninsular_pass / n
    budget_pct = 100 * budget_pass / n

    print(f"\n=== Quality metrics on {n} BJJ items ===")
    print(f"Glossary respected:    {glossary_pct:.0f}% ({glossary_pass}/{n})")
    print(f"Peninsular Spanish:    {peninsular_pct:.0f}% ({peninsular_pass}/{n})")
    print(f"Char budget compliance:{budget_pct:.0f}% ({budget_pass}/{n})")

    if failures:
        print("\nFailures:")
        for f in failures[:10]:
            print(f"  {f}")

    # Hard fail si cualquier métrica < 70%
    assert glossary_pct >= 70, f"glossary {glossary_pct:.0f}% < 70%"
    assert peninsular_pct >= 70, f"peninsular {peninsular_pct:.0f}% < 70%"
    assert budget_pct >= 70, f"budget {budget_pct:.0f}% < 70%"

    # Warning si cualquier métrica < 85%
    if min(glossary_pct, peninsular_pct, budget_pct) < 85:
        pytest.warns(UserWarning, match="quality metric below 85%")
```

- [ ] **Step 9.3.2: Verify it's skipped by default**

```bash
cd subtitle-generator && python -m pytest -v -m "not requires_ollama"
```

Expected: el test de calidad no aparece en la salida (o aparece como `SKIPPED`).

- [ ] **Step 9.3.3: Commit**

```bash
git add subtitle-generator/tests/test_translator_quality.py
git commit -m "test(quality): add automated BJJ translation quality test"
```

---

## Phase 10 — Build, smoke test, validación

### Task 10.1: Levantar Ollama y verificar pull

**Files:** ninguno (operacional).

- [ ] **Step 10.1.1: Build images**

```bash
docker compose build subtitle-generator processor-api processor-frontend
```

Expected: 3 builds exitosos.

- [ ] **Step 10.1.2: Start ollama**

```bash
docker compose up -d ollama
```

- [ ] **Step 10.1.3: Watch the pull progress**

```bash
docker compose logs -f ollama
```

Espera ver `Pulling qwen2.5:7b-instruct-q4_K_M ...` y luego `success`. Tarda ~10-15 min con red normal. Ctrl+C para salir del logs cuando termine.

- [ ] **Step 10.1.4: Verify health**

```bash
docker compose ps ollama
curl -fsS http://localhost:11434/api/tags
```

Expected: status `healthy`, JSON con `qwen2.5:7b-instruct-q4_K_M` en `models`.

---

### Task 10.2: Smoke test del translator

**Files:** ninguno.

- [ ] **Step 10.2.1: Direct translator call**

```bash
docker compose up -d subtitle-generator
docker compose exec subtitle-generator python -c "
from subtitle_generator.translator import make_translator
t = make_translator('ollama', model='qwen2.5:7b-instruct-q4_K_M')
print(t.translate_texts(['Get your grip on the sleeve and pass the guard.']))
"
```

Expected: una traducción ES con `grip` y `guard` literales conservados, registro peninsular.

- [ ] **Step 10.2.2: Run quality test against running Ollama**

```bash
cd subtitle-generator && python -m pytest tests/test_translator_quality.py -v -m requires_ollama -s
```

Expected: PASS (las 3 métricas ≥70%, ideal ≥85%). Si peninsular cae <70%, considerar tunear el prompt o subir a qwen2.5:14b.

---

### Task 10.3: Levantar resto del stack y E2E manual

**Files:** ninguno.

- [ ] **Step 10.3.1: Bring everything up**

```bash
docker compose up -d
docker compose ps
```

Expected: todos los servicios `healthy` o `running`.

- [ ] **Step 10.3.2: Open frontend**

Abre `http://localhost:5173/settings` (o el puerto de tu compose). Verifica:
- Selector "Translation Provider" muestra Ollama (default) y OpenAI.
- Sin campo DeepL.
- Dropdown de modelo muestra `qwen2.5:7b-instruct-q4_K_M`.
- Health pill muestra Ollama en verde.

- [ ] **Step 10.3.3: Process 1 BJJ episode end-to-end**

Desde el frontend, lanza el pipeline completo sobre un episodio real con `dubbing_mode=true`. Verifica:
- `.es.srt` generado con `grip`/`hook`/`underhook` literales.
- Castellano peninsular (`tú`, `coge`, `vale`).
- Char budget respetado.
- Tras translate, `nvidia-smi` muestra `qwen2.5` desaparecido en ~2 min (KEEP_ALIVE).
- Kokoro TTS arranca sin OOM en el siguiente paso.

- [ ] **Step 10.3.4: A/B vs OpenAI (recomendado)**

Cambia el provider a OpenAI en Settings, procesa el mismo episodio, compara `.es.srt` resultantes. Si Ollama es notablemente peor en >20% de líneas, decidir: bajar default a OpenAI en `_DEFAULTS` o tunear prompt.

---

### Task 10.4: Final commit + merge

- [ ] **Step 10.4.1: Verify clean state**

```bash
git status
git log --oneline -20
```

Expected: árbol limpio, ~20-25 commits incrementales en la rama.

- [ ] **Step 10.4.2: Push branch**

```bash
git push -u origin feat/ollama-translation
```

- [ ] **Step 10.4.3: Open PR via gh**

```bash
gh pr create --title "feat: ollama local translation replacing OpenAI default" --body "$(cat <<'EOF'
## Summary
- Reemplaza OpenAI gpt-4o por Ollama qwen2.5-7b local como default de traducción EN→ES.
- Elimina DeepL completamente (clase, settings, frontend, .env).
- Refactor `_BaseChatTranslator` extraído de OpenAITranslator para evitar duplicación con OllamaTranslator.
- Servicio Ollama añadido a docker-compose con pull automático.
- Test de calidad automatizable sobre golden set BJJ.

## Test plan
- [ ] Snapshot OpenAITranslator pasa pre y post refactor
- [ ] Tests OllamaTranslator (mockeados) pasan
- [ ] Build de los 3 servicios afectados (subtitle-generator, processor-api, processor-frontend)
- [ ] `docker compose up -d ollama` arranca y pull completa
- [ ] Smoke test directo del translator devuelve ES peninsular con glosario
- [ ] Test de calidad sobre golden set ≥70% en las 3 métricas
- [ ] Procesado E2E de 1 episodio BJJ con `dubbing_mode=true`
- [ ] A/B Ollama vs OpenAI sobre mismo episodio

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Plan summary

- **Total tasks:** ~30 sub-steps a través de 10 fases.
- **Commits esperados:** ~22 (uno por feature/test pair).
- **Tests añadidos:** 4 archivos (snapshot OpenAI, ollama, quality, golden set) + adaptaciones a tests existentes.
- **Archivos creados:** 4 (`ollama/entrypoint.sh`, `tests/test_translator_ollama.py`, `tests/test_translator_quality.py`, `tests/data/bjj_golden_set.jsonl`).
- **Archivos borrados:** 3 (`dubbing-generator/.../translation/translator.py`, `dubbing-generator/tests/test_translator.py`, posiblemente `__init__.py` de translation).
- **Archivos modificados:** 8 (translator.py, app.py de subtitle-generator, settings.py, pipeline.py, subtitles.py, app.py + health_proxy.py de processor-api, SettingsPage.jsx, useBackendsHealth.js, docker-compose.yml, .env.example, pyproject.toml).
- **Tiempo estimado:** ~6-10h trabajo concentrado (incluye smoke test y A/B).
- **Bloqueantes externos:** primer pull de Ollama tarda 10-15 min — planear en consecuencia.
