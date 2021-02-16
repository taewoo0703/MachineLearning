### MNIST 데이터셋
##데이터셋 내려받기
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
mnist.keys()
#배열 살펴보기
X, y = mnist["data"], mnist["target"]
X.shape
y.shape
#이미지 살펴보기
import matplotlib as mpl
import matplotlib.pyplot as plt
some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap="binary")
plt.axis("off")
plt.show()
#레이블 int로 변환
import numpy as np
y = y.astype(np.uint8)
#훈련데이터, 테스트데이터 나누기
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
