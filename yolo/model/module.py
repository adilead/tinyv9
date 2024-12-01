from typing import Any, Dict, List, Optional, Tuple, Union

from tinygrad import nn, Tensor, dtypes

from yolo.utils.module_utils import activation_func, auto_pad


class Conv:
  def __init__(self, in_channels: int, out_channels: int,
               kernel_size: int, *, activation: Optional[str] = "SiLU", **kwargs):
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, padding='same')
    self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=3e-2)
    self.act = activation_func("silu")

  def __call__(self, x: nn.Tensor) -> Tensor:
    return self.act(self.bn(self.conv(x)))


class Pool:
  def __init__(self, method: str = "max", kernel_size=2, **kwargs):
    kwargs.setdefault("padding", auto_pad(kernel_size, **kwargs))
    pools = {"max": (lambda x: x.max_pool2d(**kwargs)), "avg": (lambda x: x.avg_pool2d(**kwargs))}
    self.pool = pools[method.lower()]

  def __call__(self, x: Tensor) -> Tensor:
    return self.pool(x)


class Concat:
  def __init__(self, dim=1):
    self.dim = dim

  def __call__(self, x):
    return x[0].cat(*x[1:], dim=self.dim)


class Sequential:
  def __init__(self, *args):
    self.modules = args

  def __call__(self, x: Tensor):
    out = x
    for m in self.modules:
      out = m(out)
    return out

  def __getitem__(self, item):
    return self.modules[item]


class Detection:
  """
  YOLO Detection Head
  """

  def __init__(self, in_channels: Tuple[int], num_classes: int, *, reg_max: int = 16, use_group: bool = True):
    groups = 4 if use_group else 1
    anchor_channels = 4 * reg_max

    first_neck, in_channels = in_channels
    anchor_neck = max(round_up(first_neck // 4, groups), anchor_channels, reg_max)
    class_neck = max(first_neck, min(num_classes * 2, 128))

    self.anchor_conv = Sequential(
      Conv(in_channels, anchor_neck, 3),
      Conv(anchor_neck, anchor_neck, 3),
      nn.Conv2d(anchor_neck, anchor_channels, 1, groups=groups)
    )
    self.class_conv = Sequential(
      Conv(in_channels, class_neck, 3),
      Conv(class_neck, class_neck, 3),
      nn.Conv2d(class_neck, num_classes, 1)
    )
    self.anc2vec = Anchor2Vec(reg_max=reg_max)
    self.anchor_conv[-1].bias = Tensor.ones(self.anchor_conv[-1].bias.shape)
    self.class_conv[-1].bias = Tensor.full_like(self.class_conv[-1].bias, -10)


  def __call__(self, x: Tensor) -> Tuple[Tensor]:
    anchor_x = self.anchor_conv(x)
    class_x = self.class_conv(x)
    anchor_x, vector_x = self.anc2vec(anchor_x)
    return class_x, anchor_x, vector_x


class Anchor2Vec:
  def __init__(self, reg_max: int = 16):
    reverse_reg = Tensor.arange(reg_max, dtype=dtypes.float32).view(1, reg_max, 1, 1, 1)
    self.anc2vec = nn.Conv2d(in_channels=reg_max, out_channels=1, kernel_size=(1,1,1), bias=False) # it's a 3d conv!!
    self.anc2vec.weight = reverse_reg

  def __call__(self, anchor_x: Tensor) -> Tuple[Tensor, Tensor]:
    anchor_x = Tensor.rearrange(anchor_x, "B (P R) h w -> B R P h w", P=4)
    vector_x = anchor_x.softmax(axis=1)
    vector_x = self.anc2vec(vector_x)[:, 0]
    return anchor_x, vector_x

def round_up(x: Union[int, Tensor], div: int = 1) -> Union[int, Tensor]:
  """
  Rounds up `x` to the bigger-nearest multiple of `div`.
  """
  return x + (-x % div)