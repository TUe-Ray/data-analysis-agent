"""Deterministic helpers for the public benchmark answer contract."""

from __future__ import annotations


def validate_against_public_schema(
    value: object, schema: object, path: str = "$"
) -> None:
    """Validate the JSON-Schema subset used by public benchmark tasks."""
    if not isinstance(schema, dict):
        return
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        raise ValueError(f"{path} must be one of {enum}")
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be an object")
        required = schema.get("required", [])
        if isinstance(required, list):
            missing = [item for item in required if item not in value]
            if missing:
                raise ValueError(f"{path} is missing required field(s): {missing}")
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            properties = {}
        if schema.get("additionalProperties") is False:
            unexpected = [key for key in value if key not in properties]
            if unexpected:
                raise ValueError(f"{path} has unexpected field(s): {unexpected}")
        for key, nested_schema in properties.items():
            if key in value:
                validate_against_public_schema(
                    value[key], nested_schema, f"{path}.{key}"
                )
    elif kind == "array":
        if not isinstance(value, list):
            raise ValueError(f"{path} must be an array")
        item_schema = schema.get("items")
        for index, item in enumerate(value):
            validate_against_public_schema(item, item_schema, f"{path}[{index}]")
    elif kind == "string" and not isinstance(value, str):
        raise ValueError(f"{path} must be a string")
    elif kind == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
        raise ValueError(f"{path} must be an integer")
    elif kind == "number" and (
        not isinstance(value, (int, float)) or isinstance(value, bool)
    ):
        raise ValueError(f"{path} must be a number")


def required_schema_paths(schema: object, prefix: str = "") -> list[str]:
    """Return dotted paths for required top-level and key-result sections."""
    if not isinstance(schema, dict) or schema.get("type") != "object":
        return [prefix] if prefix else []
    if prefix.count(".") >= 1:
        return [prefix]
    required = schema.get("required", [])
    properties = schema.get("properties", {})
    if not isinstance(required, list) or not isinstance(properties, dict):
        return []
    paths: list[str] = []
    for name in required:
        if not isinstance(name, str):
            continue
        path = f"{prefix}.{name}" if prefix else name
        nested = properties.get(name)
        nested_paths = required_schema_paths(nested, path)
        paths.extend(nested_paths or [path])
    return paths
