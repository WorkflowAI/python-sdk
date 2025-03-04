from collections.abc import Collection, Mapping
from typing import Any, Optional, get_args, get_origin

from pydantic import BaseModel

from workflowai.core.utils._vars import BM


def _safe_issubclass(cls: type[Any], other: type[Any]) -> bool:
    try:
        return issubclass(cls, other)
    except TypeError:
        return False


def _construct_list(annotation: type[Any], payload: list[Any]) -> Collection[Any]:
    origin = get_origin(annotation)
    # If the type annotation is not a collection, the we can't handle it so we return as is
    if not origin or not _safe_issubclass(origin, Collection):
        return payload

    args = get_args(annotation)
    if not args:
        return payload

    constructor = set if _safe_issubclass(origin, set) else list

    if len(args) == 1 and _safe_issubclass(args[0], BaseModel):
        return constructor([construct_model_recursive(args[0], item) for item in payload])  # pyright: ignore [reportUnknownVariableType, reportCallIssue]

    return payload


def _construct_object(annotation: type[Any], payload: dict[str, Any]) -> Any:
    if _safe_issubclass(annotation, BaseModel):
        return construct_model_recursive(annotation, payload)  # pyright: ignore [reportUnknownVariableType, reportArgumentType]

    # Try to map dict of objects
    origin = get_origin(annotation)
    if not origin or not _safe_issubclass(origin, Mapping):
        return payload

    args = get_args(annotation)
    if len(args) != 2:
        return payload

    key_type, value_type = args
    if key_type is not str:
        return payload
    return {k: _construct_for_annotation(value_type, v) for k, v in payload.items()}


def _construct_for_annotation(annotation: Optional[type[Any]], payload: Any) -> Any:
    if annotation is None:
        return payload

    if isinstance(payload, dict):
        return _construct_object(annotation, payload)  # pyright: ignore [reportUnknownArgumentType]
    if isinstance(payload, list):
        return _construct_list(annotation, payload)  # pyright: ignore [reportUnknownArgumentType]

    return payload


# It does not look like there is an easy way to construct models from partial json objects
# - https://github.com/team23/pydantic-partial uses a heavy approach by constructing a new dynamic
# model class with non required fields
# - partial validation https://docs.pydantic.dev/latest/concepts/experimental/#partial-validation
# handles partial jsons but still validates that each field is present so it fails in our case
# where we just want to handle missing fields
def construct_model_recursive(model: type[BM], payload: dict[str, Any]) -> BM:
    """
    Recursively calls model construct to build a model from partial json object
    """
    mapped: dict[str, Any] = {}
    for k, v in payload.items():
        field = model.model_fields[k]
        mapped[k] = _construct_for_annotation(field.annotation, v)
    return model.model_construct(None, **mapped)
