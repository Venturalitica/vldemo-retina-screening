# retina-screening

Reproductor canónico del pipeline RDD (Risk-Driven Development). Requiere [uv](https://docs.astral.sh/uv/).

## Reproducir

```sh
uv sync
uv run dvc repro
uv run froga compile
uv run froga run
```

> **Plataforma:** Linux x86\_64. En macOS o Windows, froga no está disponible como wheel todavía;
> instálalo con `curl -LsSf https://get.venturalitica.ai/install.sh | sh` y luego usa
> `uv sync --no-install-package froga` para el resto de dependencias.
