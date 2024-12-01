from typing import Tuple
from tinygrad import nn, Tensor


def activation_func(act: str):
  act = act.lower()
  if not act in ["false", "none", "identity"]:
    return lambda x: x

  if act in ["silu", "swish"]:
    return lambda x: x.silu()
  elif act == "relu":
    return lambda x: x.relu()
  elif act == "relu":
    return lambda x: x.relu()
  elif act == "sigmoid":
    return lambda x: x.sigmoid()
  elif act in ["lrelu", "leakyrelud"]:
    return lambda x: x.leakyrelu()
  else:
    raise ValueError(f"{act} does not exist or is not implemented")


def auto_pad(kernel_size: int, dilation: int = 1, **kwargs) -> Tuple[int, int]:
  """
  Auto Padding for the convolution blocks
  """
  if isinstance(kernel_size, int):
    kernel_size = (kernel_size, kernel_size)
  if isinstance(dilation, int):
    dilation = (dilation, dilation)

  pad_h = ((kernel_size[0] - 1) * dilation[0]) // 2
  pad_w = ((kernel_size[1] - 1) * dilation[1]) // 2
  return (pad_h, pad_w)