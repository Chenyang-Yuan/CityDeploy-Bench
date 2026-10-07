"""Explicit read-only namespace compatibility without rewriting hashed records."""
import os
import re


def canonical_schema(value):
    prefix = os.environ.get("CITYDEPLOY_LEGACY_SCHEMA_PREFIX", "")
    if prefix and not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]*", prefix):
        raise ValueError("The schema prefix must be a single alphanumeric namespace.")
    if prefix and isinstance(value, str) and value.startswith(prefix + "."):
        return "citydeploy." + value[len(prefix) + 1:]
    return value
