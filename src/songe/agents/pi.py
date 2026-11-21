import copy

from .. import NAME
from ..models.names import exposed


def configure(data, config):
    data = copy.deepcopy(data)
    providers = data.setdefault("providers", {})
    existing = providers.get(NAME, {})
    from .opencode import context_for

    public_models = exposed(config)
    models = list(existing.get("models", []))
    present = {m["id"] for m in models}
    models.extend(
        {
            "id": model,
            "name": model,
            "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        }
        for model in public_models
        if model not in present
    )
    for model in models:
        if model["id"] in public_models:
            context = context_for(config, model["id"])
            model["contextWindow"] = context
            model["maxTokens"] = min(4096, context // 2)
            if model.get("compat") is None:
                model["compat"] = {}
            model["compat"].setdefault("sendSessionAffinityHeaders", True)

    providers[NAME] = {
        **existing,
        "baseUrl": f"http://127.0.0.1:{config.port}/v1",
        "api": "openai-completions",
        "apiKey": "local",
        "models": models,
    }
    return data
