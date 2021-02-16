import os
import tarfile
import urllib.request
import pandas as pd


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


# 현재 작업공간에 datasets/housing 디렉토리를 만들고 housing.tgz파일을 내려받고 같은 디렉토리에 압축을 풀어 housing.csv 파일을 생성
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# datasets/housing/housing.csv를 pandas Dataframe으로 불러옴
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

fetch_housing_data()
housing = load_housing_data()


# plot
import matplotlib.pyplot as plt

# housing.hist(bins=50, figsize=(20,15))
# plt.show()
#
#
# # random 함수로 테스트 세트 만들기
import numpy as np
#
# def split_train_test(data, test_ratio):
#     shuffled_indices = np.random.permutation(len(data))
#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled_indices[:test_set_size]
#     train_indices = shuffled_indices[test_set_size:]
#     return  data.iloc[train_indices], data.iloc[test_indices]
#
# train_set, test_set = split_train_test(housing, 0.2)
# len(train_set)
#
# # 해시값으로 테스트 세트 만들기
# from zlib import crc32
# def test_set_check(identifier, test_ratio):
#     return crc32(np.int64(identifier)) & 0xffffffff > test_ratio * 2**32
#
# def split_train_test_by_id(data, test_ratio, id_column):
#     ids = data[id_column]
#     in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
#     return data.loc[~in_test_set], data.loc[in_test_set]
#
# housing_with_id = housing.reset_index() #index를 정수로 초기화, 기존 index는 index 칼럼으로
# train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
#
# #위도와 경도로 ID 생성
# housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
# train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
#
# #사이킷런을 이용한 랜덤 샘플링
# from sklearn.model_selection import train_test_split
# train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

#사이킷런을 이용한 계층 샘플링
housing["income_cat"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1,2,3,4,5])
housing["income_cat"].hist()

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

#샘플에서 income_cat 특성 삭제
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

#훈련세트만 탐색
housing = strat_train_set.copy()

#시각화
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,\
             s=housing["population"]/100, label="population", figsize=(10,7),\
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True, sharex=False)
plt.show()

#상관관계 조사 : dataframe.corr()
corr_matrix = housing.corr()
#상관관계 조사 : 산점도
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
pd.plotting.scatter_matrix(housing[attributes], figsize=(12,8))
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
plt.show()

#가구당 인원
housing["rooms_per_household"] = housing["total_rooms"]

print('hi')










print("end")