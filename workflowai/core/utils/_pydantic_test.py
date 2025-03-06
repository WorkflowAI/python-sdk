from typing import Any, Optional

import pytest
from pydantic import BaseModel, Field, ValidationError

from workflowai.core.utils._pydantic import partial_model


class TestPartialModel:
    def test_partial_model_equals(self, recwarn: pytest.WarningsRecorder):
        class SimpleModel(BaseModel):
            name: str

        partial = partial_model(SimpleModel)
        assert partial.model_validate({"name": "John"}) == SimpleModel(name="John")

        assert SimpleModel(name="John") == partial.model_validate({"name": "John"})

        assert len(recwarn.list) == 0

    def test_simple_model(self):
        class SimpleModel(BaseModel):
            name1: str
            name2: str
            name3: int
            name4: float
            bla: dict[str, Any]
            opt: Optional[str]

        constructed = partial_model(SimpleModel).model_validate({"name1": "John"})
        assert isinstance(constructed, SimpleModel)
        assert constructed.name1 == "John"
        assert constructed.name2 == ""
        assert constructed.name3 == 0
        assert constructed.name4 == 0.0
        assert constructed.bla == {}
        assert constructed.opt is None

        # Check that we do not raise on an empty payload
        partial_model(SimpleModel).model_validate({})

        # Check that we do raise when a type is wrong
        with pytest.raises(ValidationError):
            partial_model(SimpleModel).model_validate({"name1": 1, "name2": "2"})

    def test_with_some_optional_fields(self):
        class SomeOptionalFields(BaseModel):
            name1: str
            name2: str = "blibly"
            list1: list[str] = Field(default_factory=lambda: ["1"])

        constructed = partial_model(SomeOptionalFields).model_validate({})
        assert isinstance(constructed, SomeOptionalFields)
        assert constructed.name1 == ""
        assert constructed.name2 == "blibly"
        assert constructed.list1 == ["1"]

    def test_list_of_strings(self):
        class ListOfStrings(BaseModel):
            strings: list[str]

        constructed = partial_model(ListOfStrings).model_validate({"strings": ["a", "b"]})
        assert isinstance(constructed, ListOfStrings)
        assert constructed.strings == ["a", "b"]

        # Check that we do not raise on an empty payload
        partial_model(ListOfStrings).model_validate({})

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

        constructed = partial_model(OuterModel).model_validate(payload)
        assert isinstance(constructed, OuterModel), "constructed is not an instance of OuterModel"
        assert constructed.field1 == payload.get("field1", "")
        assert isinstance(constructed.nested, NestedModel), "nested is not an instance of NestedModel"

        assert constructed.nested.name == payload.get("nested", {}).get("name", "")
        assert constructed.nested.field2 == payload.get("nested", {}).get("field2", "")

    def test_list_of_models(self):
        class NestedModel(BaseModel):
            name: str
            field2: str

        class ListOfModels(BaseModel):
            models: list[NestedModel]

        constructed = partial_model(ListOfModels).model_validate(
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

        constructed = partial_model(SetOfModels).model_validate(
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

        constructed = partial_model(DictOfModels).model_validate(
            {"models": {"hello": {"name": "hello", "field2": "world"}, "hello2": {"name": "hello"}}},
        )
        assert isinstance(constructed, DictOfModels)
        assert isinstance(constructed.models, dict)
        assert isinstance(constructed.models["hello"], NestedModel)
        assert isinstance(constructed.models["hello2"], NestedModel)

    def test_with_aliases(self):
        class AliasModel(BaseModel):
            message: str = Field(alias="message_alias")
            aliased_ser: str = Field(serialization_alias="aliased_ser_alias")
            aliased_val: str = Field(validation_alias="aliased_val_alias")

        partial = partial_model(AliasModel)

        payload = {"message_alias": "hello", "aliased_ser": "world", "aliased_val_alias": "!"}
        v1 = AliasModel.model_validate(payload)
        v2 = partial.model_validate(payload)

        assert v1 == v2
