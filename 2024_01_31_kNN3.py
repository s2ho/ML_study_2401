# DATA PREPROCESSING
# using standard score

# fish raw data

fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# data handling w/ numpy
# 기존에는 zip 함수를 써서 length / weight를 뽑아서 2D list를 만들었다.
# 이제 numpy로 처리를 해보자

import numpy as np

# numpy function : column_stack()
# 인자로 전달받은 tuple 데이터를 차례대로 나란히 연결
# ex1)
# np.column_stack(([1,2,3], [4,5,6]))
# array([[1, 4],
#        [2, 5],
#        [3, 6]])
# ex2)
# np.column_stack(([1,2,3], [4,5,6], [7,8,9]))
# array([[1, 4, 7],
#        [2, 5, 8],
#        [3, 6, 9]])

fish_data = np.column_stack((fish_length, fish_weight))

# np.zeros(a), np.ones(a)
# a 만큼의 0으로 구성된 list, 1로 구성된 list를 만들어줌.

# np.concatenate() column_stack 처럼 튜플을 인자로 받음.
# column stack과는 다르게 같은 차원으로 묶어줌. ex) np.concatenate(([1,1], [0,0])) --> [1,1,0,0]

fish_target = np.concatenate((np.ones(35), np.zeros(14)))

# print(fish_target)

# 데이터의 크기가 커질수록 python list로 data를 다루는 것이 비효율적이다.
# C, C++ 같은 low_level 언어로 개발된 numpy 배열을 사용하여 다루는 것이 빠르고 최적화되어 있다.

# 2-1의 예제에서는 배열을 직접 인덱싱하여 train / test set으로 구분하였다.
# 이건 사실 불편한 방식임.
# scikit-learn의 model_selection 모듈 에는 train_test_split() 함수가 있다.

from sklearn.model_selection import train_test_split

# 책과 같게 하기 위해 train_test_split 함수의 random_state parameter를 42로 지정
# 이 함수는 기본적으로 25%를 test set으로 떼어냄.
# train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state = 42)

# 잘 떼어냈는지 확인하기 위해 numpy의 shape(튜플 배열 모양) 함수로 확인
# 튜플의 원소가 하나면 (36,) 이런 식으로 출력됨.
# print(train_input.shape, test_input.shape, train_target.shape, test_target.shape)
# (36, 2) (13, 2) (36,) (13,)

# print(test_target)
# [1. 0. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
# 도미 10개, 빙어 3개 --> 3.3:1 --> sampling bias occurs
# 이처럼 무작위로 섞을 경우에 제대로 안 섞일 가능성 있음.

# 이걸 방지하기 위해 train_test_split() 함수의 parameter 중에 stratify 라는게 있음.
# stratify parameter에 타깃 데이터를 전달하면 클래스 비율에 맞게 데이터를 나눠줌.
# 훈련 데이터가 작거나, 특정 클래스의 샘플 개수가 적을 때 사용하면 유용

train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, stratify=fish_target, random_state=42)
# print(test_target)
# [0. 0. 1. 0. 1. 0. 1. 1. 1. 1. 1. 1. 1.] --> 2.25:1

# Now data ready
# let's train our model
# kNN model is just save train data in fit

from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
kn.score(test_input, test_target)

print(kn.predict([[25,150]])) # [0.] --> wrong answer

#let's check the data w/ plot

import matplotlib.pyplot as plt
# plt.scatter(train_input[:,0], train_input[:,1])
# plt.scatter(25,150,marker='^') #marker parameter는 plot에 찍히는 모양을 지정
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()

# 결과를 보면 도미에 더 가까운데 왜 도미가 아니라 빙어라고 예측했을까?
# kNN algorithm은 주변 샘플 중에서 다수인 클래스를 예측으로 사용한다.
# (25, 150)의 주변 샘플을 한 번 알아보자

# kneighbors() method --> 샘플에서 가까운 이웃 샘플을 찾아주는 메서드
# 거리와 샘플 인덱스를 반환함.
# n_neighbors의 기본값이 5 이므로, 5개의 이웃이 반환됨.

distances, indexes = kn.kneighbors([[25,150]])

plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25,150,marker='^')
plt.scatter(train_input[indexes,0], train_input[indexes,1], marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# --> 한개의 데이터만 도미고 나머지는 다 빙어임을 알 수 있다.
# cf. matplotlib의 marker 리스트는 bit.ly/matplotlib_marker를 참고!

#print(train_input[indexes])
#print(train_target[indexes])

print(distances) # [[ 92.00086956 130.48375378 130.73859415 138.32150953 138.39320793]]
# plot에서 나타난 거리 비율이 이상하다.x축은 범위가 좁고, y축은 범위가 넓음.
# x축의 범위를 동일하게 0~1,000으로 맞춰보자

plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25,150,marker='^')
plt.scatter(train_input[indexes,0], train_input[indexes,1], marker='D')
plt.xlim((0, 1000))
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 데이터를 표현하는 기준이 다르면 알고리즘이 올바르게 예측할 수 없음.
# 알고리즘이 거리 기반일 경우 더 그렇다. esp. kNN
# 이걸 제대로 처리하려면 특성값을 일정한 기준으로 맞춰줘야함 --> 이걸 데이터 전처리(data preprocessing) 라고 함.
# cf. 모든 알고리즘이 거리 기반인건 아님. tree 기반 알고리즘들은 feature scale이 다르더라도 제대로 동작함.

# 가장 널리 사용하는 전처리 방법은 표준점수(standard score)임.
# 각 특성 값이 평균에서 표준편차의 몇 배 만큼 떨어져 있는지를 나타내는 지표
# 분산 = {(data_0 - avg.)^2 + ... + (data_n-1 - avg.)^2}/n
# 표준편차 = 분산^(1/2)

# numpy에는 평균/표준편차를 구해주는 함수가 있다.
mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)

# train_input은 (36,2)의 배열. 특성마다 값의 스케일이 다르므로, 평균과 표준편차를 각 특성별로 계산해야함. --> axis = 0 : 행을 따라 열의 통계를 계산]
# axis = 1 이면 열을 따라 행의 통계를 계산

# print(mean, std)

train_scaled = (train_input - mean) / std
# numpy의 broadcasting 기술로 인해 train_input 배열의 모든 원소에서 위 식이 동작함.

# predict sample에도 똑같이 data 전처리를 해줘야함
# 이때 train data의 mean/std 값을 사용
new = ([25,150] - mean) / std

plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

kn.fit(train_scaled, train_target)

test_scaled = (test_input - mean)/std

kn.score(test_scaled, test_target)

distances, indexes = kn.kneighbors([new])
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker='^')
plt.scatter(train_scaled[indexes,0], train_scaled[indexes,1], marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 표준점수는 가장 널리 쓰는 scale 조정 방법이다.
# 중요한 것은 데이터 전처리 시 train set을 변환한 방식 그대로 test set도 변환해야한다는 것!