#!/usr/bin/env python3
"""生成 Acfun 二进制模型 (ACFUN0 格式) 用于端到端基准。
网络: Conv3x3(3->16,s1) -> ReLU -> Conv3x3(16->32,s2) -> ReLU
      -> MaxPool2x2 -> BN(32) -> FC(2048->10) -> Softmax
权重为随机值，仅用于性能与流程验证，不具训练意义。
"""
import struct, random, sys

random.seed(7)
MAGIC = b"ACFUN0"

def u32(x): return struct.pack("<I", x)
def f32(x): return struct.pack("<f", x)
def fblob(arr): return b"".join(f32(x) for x in arr)

def conv(prevC, Cout, KH=3, KW=3, stride=1, pad=1, name=""):
    w = [random.uniform(-0.1, 0.1) for _ in range(KH*KW*prevC*Cout)]
    b = [0.0]*Cout
    body = u32(1) + u32(KH)+u32(KW)+u32(Cout)+u32(stride)+u32(pad) + fblob(w) + fblob(b)
    return body, Cout

def fc(inDim, Cout):
    w = [random.uniform(-0.1, 0.1) for _ in range(inDim*Cout)]
    b = [0.0]*Cout
    return u32(2) + u32(Cout) + u32(inDim) + fblob(w) + fblob(b), Cout

def relu(): return u32(3)
def maxpool(pool=2, stride=2, pad=0): return u32(4) + u32(pool)+u32(stride)+u32(pad)
def softmax(): return u32(5)
def bn(ch):
    mean=[0.0]*ch; var=[1.0]*ch; gamma=[1.0]*ch; beta=[0.0]*ch
    return u32(6) + u32(ch) + fblob(mean)+fblob(var)+fblob(gamma)+fblob(beta)

def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "model.acfun"
    inC, inH, inW = 3, 32, 32
    layers = b""
    # Conv1: 3x3, stride1, pad1
    body, C = conv(inC, 16, KH=3, KW=3, stride=1, pad=1); layers += body
    layers += relu()
    # Conv2: 3x3, stride2, pad1
    body, C = conv(C, 32, KH=3, KW=3, stride=2, pad=1); layers += body
    layers += relu()
    # MaxPool
    layers += maxpool(2, 2, 0)
    # BN
    layers += bn(32)
    # FC
    flatten = 8*8*32  # 经 Conv2(s2)+MaxPool(2) 后的空间展开
    body, C = fc(flatten, 10); layers += body
    layers += softmax()

    blob = (MAGIC + u32(1)                       # version
            + u32(inC)+u32(inH)+u32(inW)
            + u32(8)                             # num_layers
            + layers)
    with open(out, "wb") as f:
        f.write(blob)
    print(f"模型已写出: {out}  ({len(blob)} bytes, 输入 {inC}x{inH}x{inW})")

if __name__ == "__main__":
    main()
