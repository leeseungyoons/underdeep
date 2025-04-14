# step24.py

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable

# Beale function 정의 (p.196)
def beale(x, y):
    z = (1.5 - x + x*y)**2 \
        + (2.25 - x + x*y**2)**2 \
        + (2.625 - x + x*y**3)**2
    return z

# ✅ (x, y) = (1.0, 0.0)에서의 gradient 구하기
x = Variable(np.array(1.0))
y = Variable(np.array(0.0))

# Beale 함수 계산 및 역전파
z = beale(x, y)
z.backward()

# 결과 출력
print("Beale function value at (1, 0):", z)
print("x.grad:", x.grad)
print("y.grad:", y.grad)
