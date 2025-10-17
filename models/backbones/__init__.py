"""
Writen by: ian
"""
from utils.registry import Registry

BACKBONES_REGISTRY = Registry(
    registry_name="Backbone_Registry",
    component_dir=["models/backbones"],
)