# Date     : 2024-01-26
# Comment  : How to handle data

# 2-1. 데이터 다루기
# 지도 학습 (supervised learning) 인 kNN에 train set과 test set을 다르게 하여 train 시켜보고 test 해보기

# fish feature data
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# make feature data into 2D list
fish_data = [[l, w] for l,w in zip(fish_length, fish_weight)] # --> There are 49 samples which each has 2 features(length & weight)

# make key data (target)
fish_target = [1] * 35 + [0] * 14

# import scikit-learn kNN & make model object
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()

#print(fish_data[0])     # [25.4, 242.0]
#print(fish_data[0:2])   # [[25.4, 242.0], [26.3, 290.0]]

# slice data and make train set & test set
train_input     = fish_data[:35]
train_target    = fish_target[:35]

test_input      = fish_data[35:]
test_target     = fish_target[35:]

# Let's train & test
kn.fit(train_input, train_target)
kn.score(test_input, test_target) # --> 0.0

# why 0.0? --> train set is about bream but, test set is about smelt
# this is called "Sampling Bias"(샘플링 편향)
# train set & test set should have separate evenly
# To make this work easier, we will use "numpy"

# Numpy
# numpy is a array library in python
# we can make and handle high dimensional data w/ numpy

import numpy as np

input_arr   = np.array(fish_data)
target_arr  = np.array(fish_target)

#print(input_arr)
#print(input_arr.shape) # (49, 2) : number of sample : number of feature

# arange() 함수 : index 만들기 + shuffle : 섞기

np.random.seed(42)          # np의 random 함수의 seed 값을 지정해주면 그 seed에 해당하는 random성으로 고정 / 아니면 계속 결과가 다름
index = np.arange(49)       # arange 함수를 쓰면 index에 0~48까지의 숫자 list 생성
np.random.shuffle(index)    # index list를 input으로 섞은 data를 input에 다시 할당하는 식인가봄

#print(index)

# Numpy : 배열 인덱싱
# print(input_arr[[1,3]]) # --> input_arr의 원소 [1], [3]을 출력

train_input     = input_arr[index[:35]] # index list의 0~35에 해당하는 숫자 원소를 train_input에 할당
train_target    = target_arr[index[:35]] # 이 예제에서는 input과 target은 항상 같이 가야함 --> supervised learning (지도 학습)이기 때문

#print(input_arr[13], train_input[0])

test_input      = input_arr[index[35:]]
test_target     = target_arr[index[35:]]

import matplotlib.pyplot as plt

# 2차원 배열은 행 index와 열 index를 ,로 나눠서 지정한다.
# 아래에 보이는 train_input[:,0]의 경우
# train_input의 전체 행 + 첫 번째 열(length) 을 의미하는 것.

plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(test_input[:,0], test_input[:,1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()


# Let's train k-Nearest Neighbor Algorithm model

kn.fit(train_input, train_target)
kn.score(test_input, test_target) # 1.0

kn.predict(test_input) # array([0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0])
# predict의 출력이 array()로 감싸져 있음을 알 수 있다
# 즉, predict() method가 반환하는 값은 단순 python list가 아니라 numpy array임.

# 정리 
# 1. 훈련에 참여하지 않은 sample을 사용해야한다.
# 2. sampling bias를 피해야 한다.
# 3. 다차원 배열 라이브러리 인 numpy를 사용했다.
# 4. numpy의 random method의 shuffle 함수를 사용하여 index를 섞었음.


# 참고
# numpy 패키지의 함수
# seed()    : 난수 생성을 위한 정수 초기값 설정
# arange()  : 일정한 간격의 정수 또는 실수 배열을 만듦, default 간격은 1
#   np.arange(3)              // [0, 1, 2]
#   np.arange(1, 3)           // [1 ,2]
#   np.arange(1, 3, 0.2)      // [1. , 1.2, 1.4, 1.6, 1.8, 2., 2.2, 2.4, 2.6, 2.8]
# shuffle() : 주어진 배열을 랜덤하게 섞음. 다차원 배열의 경우 첫 번째 축(행) 만 섞음.
#   arr = np.array([[1,2], [3,4], [5,6]])
#   np.random.shuffle(arr) 
#   ex) --> [[3 4]
#            [5 6]
#            [1 2]]   