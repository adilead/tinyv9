from typing import Union, Dict, List, Optional
from pathlib import Path
import yaml

from tinygrad.tensor import Tensor
from tinygrad.nn.state import safe_load, load_state_dict

from yolo.config.config import ModelConfig

class YOLO:

  def __init__(self, model_cfg: ModelConfig, class_num: int):
    pass

  def __call__(self, x: Tensor):
    return x

  def save_load_weights(self, weights: Path):
    """
    Update the model's weights with the provided weights.

    args:
        weights: A OrderedDict containing the new weights.
    """
    if isinstance(weights, Path):
      state_dict = safe_load(weights)
    # if "model_state_dict" in weights:
    #   state_dict = weights["model_state_dict"]

    # model_state_dict = self.model.state_dict()
    load_state_dict(self, state_dict)

    # TODO1: autoload old version weight
    # TODO2: weight transform if num_class difference

    # TODO error handling
    # error_dict = {"Mismatch": set(), "Not Found": set()}
    # for model_key, model_weight in model_state_dict.items():
    #   if model_key not in weights:
    #     error_dict["Not Found"].add(tuple(model_key.split(".")[:-2]))
    #     continue
    #   if model_weight.shape != weights[model_key].shape:
    #     error_dict["Mismatch"].add(tuple(model_key.split(".")[:-2]))
    #     continue
    #   model_state_dict[model_key] = weights[model_key]
    #
    # for error_name, error_set in error_dict.items():
    #   for weight_name in error_set:
    #     print(f":warning: Weight {error_name} for key: {'.'.join(weight_name)}")
    #
    # self.model.load_state_dict(model_state_dict)


def create_model(config_path: str, weight_path: Union[bool, Path] = True, class_num: int = 80) -> YOLO:
  with open(config_path) as f:
    model_cfg = ModelConfig(yaml.safe_load(f))
    model = YOLO(model_cfg, class_num)
    if weight_path:
      if weight_path == True:
        weight_path = Path("weights") / f"{model_cfg.name}.pt"
      elif isinstance(weight_path, str):
        weight_path = Path(weight_path)

      if not weight_path.exists():
        print(f"🌐 Weight {weight_path} not found, try downloading")
        prepare_weight(weight_path=weight_path)
      if weight_path.exists():
        model.save_load_weights(weight_path)
        print(":white_check_mark: Success load model & weight")
    else:
      print(":white_check_mark: Success load model")
    return model

def prepare_weight(download_link: Optional[str] = None, weight_path: Path = Path("v9-c.pt")):
  weight_name = weight_path.name
  if download_link is None:
    download_link = "https://github.com/WongKinYiu/yolov9mit/releases/download/v1.0-alpha/"
  weight_link = f"{download_link}{weight_name}"

  if not weight_path.parent.is_dir():
    weight_path.parent.mkdir(parents=True, exist_ok=True)

