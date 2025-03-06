import warnings
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar, get_args, get_origin

from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined
from typing_extensions import TypeGuard

from workflowai.core._logger import logger
from workflowai.core.utils._vars import BM

_T = TypeVar("_T")


def _safe_issubclass(val: type[_T], cls: type[_T]) -> TypeGuard[type[_T]]:
    try:
        return issubclass(val, cls)
    except TypeError:
        return False


def _copy_field_info(field_info: FieldInfo, **overrides: Any):
    """
    Return a copy of a pydantic FieldInfo object, allow to override
    certain values.
    """

    _excluded = {"annotation", "required"}

    kwargs = overrides
    for k, v in field_info.__repr_args__():
        if k in kwargs or not k or k in _excluded:
            continue
        kwargs[k] = v

    return Field(**kwargs)


def _default_value_from_annotation(annotation: type[Any]) -> Any:
    try:
        # Trying to see if the object is instantiable with no value
        return annotation()
    except Exception:  # noqa: BLE001
        logger.debug("Failed to get default value from annotation", exc_info=True)
    # Fallback to None
    return None


def _optional_annotation(annotation: type[Any]) -> type[Any]:
    if _safe_issubclass(annotation, BaseModel):
        return partial_model(annotation)

    origin = get_origin(annotation)
    args = get_args(annotation)
    if not origin or not args:
        return annotation

    if _safe_issubclass(origin, Sequence) or _safe_issubclass(origin, set):
        if not len(args) == 1:
            raise ValueError("Sequence must have exactly one argument")
        return origin[_optional_annotation(args[0])]
        # No need to do anything here ?

    if _safe_issubclass(origin, Mapping):
        if not len(args) == 2:
            raise ValueError("Mapping must have exactly two arguments")
        if args[0] is not str:
            raise ValueError("Mapping key must be a string")

        return origin[args[0], _optional_annotation(args[1])]
    return annotation


def partial_model(base: type[BM]) -> type[BM]:
    default_fields: dict[str, tuple[type[Any], FieldInfo]] = {}
    for name, field in base.model_fields.items():
        if field.default != PydanticUndefined or field.default_factory or not field.annotation:
            # No need to do anything here, the field is already optional
            continue

        overrides: dict[str, Any] = {}
        try:
            annotation = _optional_annotation(field.annotation)
            overrides["default"] = _default_value_from_annotation(annotation)
        except Exception:  # noqa: BLE001
            logger.debug("Failed to make annotation optional", exc_info=True)
            continue

        default_fields[name] = (annotation, _copy_field_info(field, **overrides))

    if not default_fields:
        return base

    def custom_eq(o1: BM, o2: Any):
        if not isinstance(o2, base):
            return False
        return o1.model_dump() == o2.model_dump()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="fields may not start with an underscore")
        return create_model(  # pyright: ignore [reportCallIssue, reportUnknownVariableType]
            f"Partial{base.__name__}",
            __base__=base,
            __eq__=custom_eq,
            __hash__=base.__hash__,
            **default_fields,  # pyright: ignore [reportArgumentType]
        )
