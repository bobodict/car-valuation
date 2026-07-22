# services/__init__.py
# 方便其它地方直接 from services import call_model_api

__all__ = ["call_model_api"]


def __getattr__(name: str):
    if name == "call_model_api":
        from .model_service import call_model_api

        return call_model_api
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
