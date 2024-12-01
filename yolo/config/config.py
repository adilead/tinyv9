from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Union
import yaml


@dataclass
class AnchorConfig:
  strides: List
  anchor: List
  reg_max: Optional[int] = None
  anchor_num: Optional[int] = None


@dataclass
class LayerConfg:
  args: Dict
  source: Union[int, str, List[int]]
  tags: str


@dataclass
class BlockConfig:
  block: List[Dict[str, LayerConfg]]


@dataclass
class ModelConfig:
  name: Optional[str]
  anchor: AnchorConfig
  model: Dict[str, BlockConfig]


def load_yaml(file_path: str):
  with open(file_path, 'r') as file:
    return yaml.safe_load(file)


def dict_to_dataclass(cls, d):
  """Convert a dictionary to a dataclass."""
  field_names = {f.name for f in fields(cls)}
  filtered_dict = {k: v for k, v in d.items() if k in field_names}

  for field in fields(cls):
    if field.type in (int, float):
      filtered_dict[field.name] = filtered_dict.get(field.name, 0)
    elif field.type in (str,):
      filtered_dict[field.name] = filtered_dict.get(field.name, "")
    elif field.type in (List,):
      filtered_dict[field.name] = filtered_dict.get(field.name, [])
    if hasattr(field.type, '__dataclass_fields__'):
      # Recursively convert nested dataclasses
      filtered_dict[field.name] = dict_to_dataclass(field.type, filtered_dict[field.name])

  return cls(**filtered_dict)