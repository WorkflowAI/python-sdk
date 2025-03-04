from typing import Any

import pytest
from pydantic import BaseModel

from workflowai.core.utils._pydantic import construct_model_recursive


class TestConstructModelRecursive:
    def test_simple_model(self):
        class SimpleModel(BaseModel):
            name1: str
            name2: str

        constructed = construct_model_recursive(SimpleModel, {"name1": "John"})
        assert isinstance(constructed, SimpleModel)
        assert constructed.name1 == "John"
        with pytest.raises(AttributeError):
            _ = constructed.name2

    def test_list_of_strings(self):
        class ListOfStrings(BaseModel):
            strings: list[str]

        constructed = construct_model_recursive(ListOfStrings, {"strings": ["a", "b"]})
        assert isinstance(constructed, ListOfStrings)
        assert constructed.strings == ["a", "b"]

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"field1": "hello"},
            {"nested": {"name": "hello", "field2": "world"}},
            {"nested": {"name": "hello"}},
        ],
    )
    def test_nested_model(self, payload: dict[str, Any]):
        class NestedModel(BaseModel):
            name: str
            field2: str

        class OuterModel(BaseModel):
            field1: str
            nested: NestedModel

        constructed = construct_model_recursive(OuterModel, payload)
        assert isinstance(constructed, OuterModel), "constructed is not an instance of OuterModel"
        if "field1" in payload:
            assert isinstance(constructed.field1, str), "field1 is not a string"
        else:
            with pytest.raises(AttributeError):
                _ = constructed.field1

        if "nested" in payload:
            assert isinstance(constructed.nested, NestedModel), "nested is not an instance of NestedModel"
            if "name" in payload["nested"]:
                assert isinstance(constructed.nested.name, str)
            else:
                with pytest.raises(AttributeError):
                    _ = constructed.nested.name

            if "field2" in payload["nested"]:
                assert isinstance(constructed.nested.field2, str)
            else:
                with pytest.raises(AttributeError):
                    _ = constructed.nested.field2
        else:
            with pytest.raises(AttributeError):
                _ = constructed.nested

    def test_list_of_models(self):
        class NestedModel(BaseModel):
            name: str
            field2: str

        class ListOfModels(BaseModel):
            models: list[NestedModel]

        constructed = construct_model_recursive(
            ListOfModels,
            {"models": [{"name": "hello", "field2": "world"}, {"name": "hello"}]},
        )
        assert isinstance(constructed, ListOfModels)
        assert isinstance(constructed.models, list)
        assert isinstance(constructed.models[0], NestedModel)
        assert isinstance(constructed.models[1], NestedModel)

    def test_set_of_models(self):
        class NestedModel(BaseModel):
            name: str = "1"
            field2: str

            def __hash__(self) -> int:
                return hash(self.name)

        class SetOfModels(BaseModel):
            models: set[NestedModel]

        constructed = construct_model_recursive(
            SetOfModels,
            {"models": [{"name": "hello", "field2": "world"}, {"name": "hello"}]},
        )
        assert isinstance(constructed, SetOfModels)
        assert isinstance(constructed.models, set)
        assert all(isinstance(model, NestedModel) for model in constructed.models)

    def test_dict_of_models(self):
        class NestedModel(BaseModel):
            name: str
            field2: str

        class DictOfModels(BaseModel):
            models: dict[str, NestedModel]

        constructed = construct_model_recursive(
            DictOfModels,
            {"models": {"hello": {"name": "hello", "field2": "world"}, "hello2": {"name": "hello"}}},
        )
        assert isinstance(constructed, DictOfModels)
        assert isinstance(constructed.models, dict)
        assert isinstance(constructed.models["hello"], NestedModel)
        assert isinstance(constructed.models["hello2"], NestedModel)
