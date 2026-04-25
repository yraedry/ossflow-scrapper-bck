Plataforma de procesamiento de instruccionales de BJJ (v2). Componentes independientes (SOLID, DRY). Settings dinámicos vía web, Docker desacoplado de rutas hardcodeadas.

Estructura
C:\proyectos
python
bjj-base/ Imagen Docker base compartida (CUDA 12.4 + torch + FastAPI) bjj_service_kit/ Paquete compartido: app factory, ring-buffer logs, /health, /gpu, /logs chapter-splitter/ Fragmentación en capítulos (oracle / signal) subtitle-generator/ WhisperX → subtítulos EN dubbing-generator/ Coqui XTTS v2 → doblaje ES processor-api/ FastAPI gateway: orquesta backends, settings, pipeline, jobs processor-frontend/ React 18 + Tailwind + Zustand (SPA) telegram-fetcher/

[INSTRUCCIÓN ESTRICTA PARA EL AGENTE]
Para ahorrar contexto, NO busques ni leas archivos al azar. Lee ÚNICAMENTE el documento que corresponda estrictamente a tu tarea actual usando la herramienta de lectura:

Para Frontend / UI: Lee docs/contexto_frontend.md
Para Backend / API / Pipeline: Lee docs/contexto_backend.md
Para Oráculo / BJJFanatics: Lee docs/contexto_oracle.md
Para Infra / Docker / Despliegue: Lee docs/contexto_infra.md
Convenciones al añadir features
Agentes en paralelo: para features disjuntas, dispatch con tareas claramente acotadas a archivos no solapados. Backend + frontend en agentes separados.
Nuevo endpoint: crear router en api/<feature>.py, tests en tests/test_<feature>.py, wiring en api/app.py.
Nueva página: crear en src/pages/<Name>Page.jsx + test + añadir ruta en App.jsx + nav item en Sidebar.jsx.
No tocar api/app.py, App.jsx, Layout.jsx, Sidebar.jsx en agentes paralelos — dejar líneas WIRE_* en el reporte para hacer wiring central sin conflictos.

## Approach
- Think before acting. Read existing files before writing code.
- Be concise in output but thorough in reasoning.
- Prefer editing over rewriting whole files.
- Do not re-read files you have already read unless the file may have changed.
- Skip files over 100KB unless explicitly required.
- Suggest running /cost when a session is running long to monitor cache ratio.
- Recommend starting a new session when switching to an unrelated task.
- Test your code before declaring done.
- No sycophantic openers or closing fluff.
- Keep solutions simple and direct.
- User instructions always override this file.
