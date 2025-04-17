if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
import dezero
if not dezero.is_simple_core:
    from dezero.core_simple import Variable
    from dezero.core_simple import setup_variable
    setup_variable()

# 로젠브록 함수 정의
def rosenbrock(x0, x1):
    return 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2

# 그래디언트 클리핑 함수 (폭주 방지용)
def clip_grad(x, max_norm=1.0):
    grad = x.grad
    norm = np.sqrt(np.sum(grad ** 2))
    if norm > max_norm:
        x.grad *= max_norm / (norm + 1e-6)

# 초기값 설정
x0 = Variable(np.array(0.0))     # x0 = 0
x1 = Variable(np.array(-1.0))    # x1 = -1

# 하이퍼파라미터
lr = 0.05
iters = 10000

# 경사하강법 루프
for i in range(iters):
    y = rosenbrock(x0, x1)

    x0.cleargrad()
    x1.cleargrad()
    y.backward()

    # 폭주 방지를 위한 그래디언트 클리핑
    clip_grad(x0)
    clip_grad(x1)

    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad

    # 출력: 1000번마다
    if i % 1000 == 0 or i == iters - 1:
        print(f"iter {i:5d} | x0 = {x0.data:.6f}, x1 = {x1.data:.6f}, loss = {y.data:.8f}")

# 최종 결과 출력
print("\n--- 최솟값 근사 ---")
print(f"x0 = {x0.data}, x1 = {x1.data}")
