"""
Writen by: ian
"""
from utils.registry import Registry

ARCHITECTURES_REGISTRY = Registry(
    registry_name="Architecture_Registry",
    component_dir=["models/architectures"],
)


