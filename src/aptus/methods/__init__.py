"""Typed fine-tuning method identities and lifecycle metadata."""

from .contracts import MethodDescriptor, MethodLifecycle
from .registry import (
    METHOD_REGISTRY,
    descriptor_for_compiler,
    method_descriptor,
    method_descriptors,
    selectable_method_descriptors,
    selectable_method_ids,
)

__all__ = [
    "METHOD_REGISTRY",
    "MethodDescriptor",
    "MethodLifecycle",
    "descriptor_for_compiler",
    "method_descriptor",
    "method_descriptors",
    "selectable_method_descriptors",
    "selectable_method_ids",
]
