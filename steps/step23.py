# step23.py

# 1. dezero 경로 추가 (23.5절 방식)
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 2. 필요한 모듈 임포트
import numpy as np
from dezero import Variable

# 3. Variable 테스트 코드
x = Variable(np.array(1.0))
y = (x + 3) ** 2
y.backward()

# 4. 결과 출력
print(y)        # variable(16.0)
print(x.grad)   # 8.0
