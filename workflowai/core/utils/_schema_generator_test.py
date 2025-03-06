from pydantic import BaseModel

from workflowai.core.utils._schema_generator import JsonSchemaGenerator


class TestJsonSchemaGenerator:
    def test_generate(self):
        class TestModel(BaseModel):
            name: str

        schema = TestModel.model_json_schema(schema_generator=JsonSchemaGenerator)
        assert schema == {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}

    def test_nested_model(self):
        class NestedModel(BaseModel):
            name: str

        class TestModel(BaseModel):
            nested: NestedModel

        schema = TestModel.model_json_schema(schema_generator=JsonSchemaGenerator)
        assert schema == {
            "$defs": {
                "NestedModel": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            "type": "object",
            "properties": {
                "nested": {"$ref": "#/$defs/NestedModel"},
            },
            "required": ["nested"],
        }
