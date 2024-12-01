import sys
from pathlib import Path

import numpy as np
from tinygrad import Tensor
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from yolo.model.module import Conv, Pool, Concat, Sequential, Detection

STRIDE = 2
KERNEL_SIZE = 3
IN_CHANNELS = 64
OUT_CHANNELS = 128
NECK_CHANNELS = 64


def test_conv():
  conv = Conv(IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE)
  x = Tensor.randn(1, IN_CHANNELS, 64, 64)
  out = conv(x)
  assert out.shape == (1, OUT_CHANNELS, 64, 64)


def test_pool_max():
  pool = Pool("max", 2, stride=2)
  x = Tensor.randn(1, IN_CHANNELS, 64, 64)
  out = pool(x)
  assert out.shape == (1, IN_CHANNELS, 32, 32)


def test_pool_avg():
  pool = Pool("avg", 2, stride=2)
  x = Tensor.randn(1, IN_CHANNELS, 64, 64)
  out = pool(x)
  assert out.shape == (1, IN_CHANNELS, 32, 32)


def test_concat():
  concat = Concat(dim=1)
  x = [Tensor.ones(2,3), Tensor.ones(2,4), Tensor.ones(2,2)]
  out = concat(x)
  assert out.shape == (2,9)


def test_sequential():
  seq = Sequential(
    lambda x: x * 2,
    lambda x: x + 1,
  )
  res = seq(Tensor.ones(1,2))
  assert res.shape == (1,2)
  assert np.array_equal(res.numpy(), np.array([[3.0,3.0]]))


def test_Detection():
  detection = Detection((23,32), 80)
  x = Tensor.randn((4,32, 100, 100))
  klass, anch, vec = detection(x)
  assert klass.shape == (4,80,100,100)
  assert anch.shape == (4,16,4,100,100)
  assert vec.shape == (4,4,100,100)


# def test_adown():
#     adown = ADown(IN_CHANNELS, OUT_CHANNELS)
#     x = torch.randn(1, IN_CHANNELS, 64, 64)
#     out = adown(x)
#     assert out.shape == (1, OUT_CHANNELS, 32, 32)
#
#
# def test_cblinear():
#     cblinear = CBLinear(IN_CHANNELS, [5, 5])
#     x = torch.randn(1, IN_CHANNELS, 64, 64)
#     outs = cblinear(x)
#     assert len(outs) == 2
#     assert outs[0].shape == (1, 5, 64, 64)
#     assert outs[1].shape == (1, 5, 64, 64)
#
#
# def test_sppelan():
#     sppelan = SPPELAN(IN_CHANNELS, OUT_CHANNELS, NECK_CHANNELS)
#     x = torch.randn(1, IN_CHANNELS, 64, 64)
#     out = sppelan(x)
#     assert out.shape == (1, OUT_CHANNELS, 64, 64)