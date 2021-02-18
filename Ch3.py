### MNIST 데이터셋
## 데이터셋 내려받기
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

### 2진 분류기 훈련
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
## 확률적 경사하강법
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

sgd_clf.predict([some_digit])

### 성능 측정
## 교차검증
#교차검증 만들기 폴드 3개짜리
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))

#교차검증 cross_val_score() 사용
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")

#5아님 더미분류기 생성
from sklearn.base import BaseEstimator
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        return self
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)
#accuracy 지표의 함정
never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")

## 오차행렬
#cross_val_predict() 사용
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
#confusion_matrix() 사용해 오차행렬 생성
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_train_pred)

## 정밀도와 재현율
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_5, y_train_pred) #정밀도
recall_score(y_train_5, y_train_pred) #재현율

#F1 score : 정밀도와 재현율의 조화평균
from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)

## 정밀도/재현율 트레이드오프
#decision_function()으로 각 샘플의 점수 확인
y_scores = sgd_clf.decision_function([some_digit])
y_scores
threshold = 9000
y_some_digit_pred = (y_scores > threshold)

#모든 샘플의 점수 구하기
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")
#precision_recall_curve
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="정밀도")
    plt.plot(thresholds, recalls[:-1], "g-", label="재현율")
    plt.legend(loc="center right", fontsize=16)
    plt.xlabel("Threshold", fontsize=16)
    plt.grid(True)
    plt.axis([-50000, 50000, 0, 1])

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

recall_90_precision = recalls[np.argmax(precisions >= 0.90)]
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]

plt.figure(figsize=(8, 4))                                                                  # Not shown
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.plot([threshold_90_precision, threshold_90_precision], [0., 0.9], "r:")                 # Not shown
plt.plot([-50000, threshold_90_precision], [0.9, 0.9], "r:")                                # Not shown
plt.plot([-50000, threshold_90_precision], [recall_90_precision, recall_90_precision], "r:")# Not shown
plt.plot([threshold_90_precision], [0.9], "ro")                                             # Not shown
plt.plot([threshold_90_precision], [recall_90_precision], "ro")                             # Not shown
plt.show()

#재현율-정밀도 곡선 그리기
def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.grid(True)
plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)
plt.plot([recall_90_precision, recall_90_precision], [0., 0.9], "r:")
plt.plot([0.0, recall_90_precision], [0.9, 0.9], "r:")
plt.plot([recall_90_precision], [0.9], "ro")
#정밀도 90%인 쓰레쉬홀드에 대하여 재현율 계산
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]
y_train_pred_90 = (y_scores >= threshold_90_precision)
precision_score(y_train_5, y_train_pred_90)
recall_score(y_train_5, y_train_pred_90)

## ROC 곡선 - 수신기조작특성 : FP비율에 대한 TP비율(재현율)
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
#곡선그리기
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') #대각점선
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16)
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)
    plt.grid(True)
plt.figure(figsize=(8,6))
plot_roc_curve(fpr, tpr)
fpr_90 = fpr[np.argmax(tpr >= recall_90_precision)]           # Not shown
plt.plot([fpr_90, fpr_90], [0., recall_90_precision], "r:")   # Not shown
plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], "r:")  # Not shown
plt.plot([fpr_90], [recall_90_precision], "ro")               # Not shown
plt.show()

#곡선아래 면적(AUC측정) : 완벽하면 1, 랜덤분류기는 0.5
from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)

#RandomForestClassifier 와 SGDClasifier 비교
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method="predict_proba")
y_scores_forest = y_probas_forest[:,1] #양성 클래스에 대한 확률을 점수로 사용
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)
#비교 곡선그리기
plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "랜덤 포레스트")
plt.legend(loc="lower right")
plt.show()

### 다중분류
#SVM 분류기 테스트
from sklearn.svm import SVC # SVC는 OvO전략을 사용, LinearSVC는 OvR 전략을 사용
svm_clf = SVC()
svm_clf.fit(X_train, y_train) # OvO 전략을 사용해 훈련
svm_clf.predict([some_digit])
#첫번 째 샘플로 확인
some_digit_scores = svm_clf.decision_function([some_digit])
some_digit_scores
np.argmax(some_digit_scores)
svm_clf.classes_
svm_clf.classes_[5]

#SVC기반으로 OvR전략을 사용하는 다중분류기 생성
from sklearn.multiclass import OneVsRestClassifier
ovr_clf = OneVsRestClassifier(SVC())
ovr_clf.fit(X_train, y_train)
ovr_clf.predict([some_digit])
len(ovr_clf.estimators_)

#SGDClasifier는 직접 다중 클래스 분류가능
sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])
sgd_clf.decision_function([some_digit])
# cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")

#입력의 스케일 변화로 정확도 높이기
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
# cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")

### 에러분석 - 가능성이 높은 모델을 하나 찾았다고 가정하고, 이 모델의 성능을 향상 시킬 방법 모색
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train)
conf_mx = confusion_matrix(y_train, y_train_pred)
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()
#에러 비율확인
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()

#3과 5의 샘플 그려보기
cl_a, cl_b = 3, 5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]
# 숫자 그림을 위한 추가 함수
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis("off")
plt.figure(figsize=(8,8))
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
plt.show()

### 다중 레이블 분류 - 여러 개의 이진 레이블
from sklearn.neighbors import KNeighborsClassifier
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
knn_clf.predict([some_digit])
#평가하기 F1점수 사용
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average="macro")

### 3.7 다중 출력 분류 - 다중 레이블 분류에서 한 레이블이 다중 클래스가 될 수 있도록 일반화 한 것
# 픽셀강도에 잡음 추가
noise = np.random.randint(0, 100, (len(X_train), 784)) # 60000*784 짜리 랜덤인트 행렬
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test
some_index = 0
knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[some_index]])
def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = mpl.cm.binary,
               interpolation="nearest")
    plt.axis("off")
plot_digit(clean_digit)


####################################################################################
### 3.8 연습문제
## 1번 문제
from sklearn.model_selection import GridSearchCV
param_grid = [
    {'weights':('uniform', 'distance'), 'n_neighbors':range(1,11)}
]
knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knn_clf, param_grid, cv=3, scoring="accuracy")
grid_search.fit(X_train_scaled, y_train)
grid_search.best_score_
grid_search.best_params_
grid_search.best_estimator_

## 2번 문제
from scipy.ndimage.interpolation import shift
def shift_image(digit, left=0, right=0, up=0, down=0):
    img = digit.reshape(28, 28)
    img_s = img.copy()
    if left >= right:
        for i in range(img_s.shape[1]-(left-right)):
            img_s[:, i] = img_s[:, i+(left-right)]
        img_s[:, -(left-right):] = 0
    else:
        for i in range(img_s.shape[1]-(right-left)):
            img_s[:, -(i+1)] = img_s[:, -(i+1)-(right-left)]
        img_s[:, :(right-left)] = 0

    if up >= down:
        for i in range(img_s.shape[0]-(up-down)):
            img_s[i, :] = img_s[i+(up-down), :]
        img_s[-(up-down):, :] = 0
    else:
        for i in range(img_s.shape[0]-(down-up)):
            img_s[-(i+1), :] = img_s[-(i+1)-(down-up), :]
        img_s[:(down-up), :] = 0
    return img_s

def add_shifted_image(X):
    X_added = X.copy()
    for digit in X:
        X_added = np.vstack([X_added, shift_image(digit, left=1).reshape(1, 784)])
        X_added = np.vstack([X_added, shift_image(digit, right=1).reshape(1, 784)])
        X_added = np.vstack([X_added, shift_image(digit, up=1).reshape(1, 784)])
        X_added = np.vstack([X_added, shift_image(digit, down=1).reshape(1, 784)])
    return X_added

X_train_added = add_shifted_image(X_train)
len(X_train_added)
