import yaml

from yolo.config.config import ModelConfig, dict_to_dataclass
from yolo.model.yolo import YOLO


def test_yolo_v9m():
  with open("../../configs/v9-m.yaml") as f:
    model_cfg = dict_to_dataclass(ModelConfig, yaml.safe_load(f.read()))
    model = YOLO(model_cfg, class_num=80)
