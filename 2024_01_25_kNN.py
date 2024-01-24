# Date     : 2024-01-25
# Comment  : Today I start ML/DL study!

# k-Nearest Neighbors Algorithm
# 특정 data를 predict 시 가장 가까운 직선거리에 어떤 data가 있는지 확인하여 그 값을 반환하는 알고리즘
# data가 많은 경우 사용하기 어렵다. data가 커서 메모리가 많이 필요. 직선 거리 계산 시 많은 시간 소요

# (1) bream feature data

bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

# (2) smelt feature data
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# (3) plotting
import matplotlib.pyplot as plt # matplotlib의 pyplot 함수를 plt로 줄여서 사용

plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# (4) k-Nearest Neighbor algorithm

# collect data
length = bream_length + smelt_length
weight = bream_weight + smelt_weight

# Raw data to 2D data set for scikit-learn
fish_data = [[l, w] for l, w in zip(length, weight)]

print(fish_data)

# make a key data
fish_target = [1] * 35 + [0] * 14

# import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

# assign object to a model 'kn'
kn = KNeighborsClassifier()

# model training
kn.fit(fish_data, fish_target)

# scoring --> value 0(0% correct) ~ 1(100% correct)
kn.score(fish_data, fish_target)

# test
kn.predict([[30,600], [10, 2]])

# scikit-learn : KNeighborsClassifier Class
print(kn._fit_X)
print(kn._y)

# actually there is no training in KNA
# just compare input data with _fit_X & _y
# default n_neighbors = 5 // how many reference data compare with input data

kn49 = KNeighborsClassifier(n_neighbors=49)

kn49.fit(fish_data, fish_target)
kn49.score(fish_data, fish_target)

# wrap-up test
# 최초로 1이 안되는 n_neibors는?

kn_t = KNeighborsClassifier()
kn_t.fit(fish_data, fish_target)

for n in range(5,50) :
    kn_t.n_neighbors = n
    score = kn_t.score(fish_data, fish_target)
    if score < 1 :
        print(n, score)
        break;