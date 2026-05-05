"""
JSON Schema generation for TauluConfig TOML files.

Run ``taulu-schema > taulu-config.schema.json`` to export the schema to your
project directory, then reference it in your TOML file::

    "$schema" = "./taulu-config.schema.json"

Editors with taplo support (VS Code "Even Better TOML", Neovim) will use this
for autocompletion and validation.

Multiple TOML files can be merged, with later files overriding earlier ones::

    "$schema" = "./taulu-config.schema.json"
    # common.toml — shared parameters
    binarization_sensitivity = 0.05

    # left.toml — overrides only what differs
    "$schema" = "./taulu-config.schema.json"
    template_path = "header_left.png"
"""

import json
import sys

from .config import TauluConfig


def generate_schema() -> dict:
    """
    Build a JSON Schema (draft-07) for `TauluConfig` TOML files, with an
    extra ``$schema`` property so editors can self-reference the schema.

    Returns:
        dict: the JSON Schema as a Python dictionary.
    """
    schema = TauluConfig.model_json_schema()
    schema["$schema"] = "http://json-schema.org/draft-07/schema#"
    schema["properties"]["$schema"] = {
        "type": "string",
        "description": "Path or URL to this JSON Schema file.",
    }
    return schema


def main():
    """Print the generated schema as indented JSON to stdout."""
    print(json.dumps(generate_schema(), indent=2))


if __name__ == "__main__":
    sys.exit(main())
