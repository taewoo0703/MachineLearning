### 4.1 선형회귀
## 4.1.1 정규방정식
import numpy as np
X = 2* np.random.rand(100, 1)
y = 4 + 3*X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X] #모든 샘플에 x0 = 1을 추가합니다.
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y) #정규방정식
#0과 2라는 샘플에 대한 y예측값
X_new = np.array([[0],
                  [2]])
X_new_b = np.c_[np.ones((2,1)), X_new] #모든 샘플에 x0 = 1 을 추가합니다.
y_predict = X_new_b.dot(theta_best)
#그래프로 표현
import matplotlib.pyplot as plt
plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()
