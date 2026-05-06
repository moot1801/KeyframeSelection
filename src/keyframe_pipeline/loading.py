from __future__ import annotations

import importlib
from typing import Any


def load_class(module_path: str, class_name: str) -> type[Any]:
    if not module_path:
        raise ValueError("동적 로딩에는 module 경로가 필요합니다.")
    if not class_name:
        raise ValueError("동적 로딩에는 class_name 값이 필요합니다.")

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(f"module을 import할 수 없습니다: {module_path}") from exc

    try:
        loaded = getattr(module, class_name)
    except AttributeError as exc:
        raise AttributeError(f"{module_path}에서 class를 찾을 수 없습니다: {class_name}") from exc

    if not isinstance(loaded, type):
        raise TypeError(f"{module_path}.{class_name}은 class가 아닙니다.")
    return loaded


def instantiate_class(module_path: str, class_name: str, *args: Any, **kwargs: Any) -> Any:
    loaded_class = load_class(module_path=module_path, class_name=class_name)
    try:
        return loaded_class(*args, **kwargs)
    except TypeError as exc:
        raise TypeError(
            f"{module_path}.{class_name} 인스턴스 생성에 실패했습니다. "
            "config의 module/class_name/kwargs를 확인하세요."
        ) from exc


def require_method(instance: Any, method_name: str, owner: str) -> None:
    method = getattr(instance, method_name, None)
    if not callable(method):
        raise TypeError(f"{owner}에는 호출 가능한 '{method_name}' 메서드가 필요합니다.")

