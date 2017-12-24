#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
=====
@author: qqzeng
@email: qtozeng@gmail.com
@date: 2017/12/24
=====

@desc: the program of the 18th, 19th and 20th in hw#1 of ntuml
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import hw01_15

"""
pocket PLA 其实也比较简单。其要点为：
1. 任何时候，我放在口袋里面的一定是到目前为止，最好的拟合函数 g。
2. 为什么说 g 是最好的，因为它在训练集上的错误率是最小的。
3. 每更新一次权重向量，就要测试其在训练集上的错误率，以确定是否要将我口袋里面进行替换掉。
那么，它和 PLA 有什么异同呢？
1.相同点：二者更新权重的方式类似。
2.不同点: PLA 其实并不能保证在任何时候，其拿到的权重向量(也可以认为是g)是最好的。
它只是不断的测试样本，然后从所有出错的样本当中随机选出一个作为更新权重的样本。
换言之，修正权重向量在这个样本上的错误，这样就把权重向更接近目标函数(f)方向进行更新。
更新后，需要比较此时得到的新的权重向量在训练集上的错误率是否比我口袋里面的要好，
如果要好，我就把我口袋里面保存的进行替换，否则，继续从所有出错的样本中随机选一个样本进行更新。
"""

# acquire the mean error on test data
def test_error(x, y, w):
    return np.sum(y != hw01_15.sign(x.dot(w))) / len(y)


def PLA_pocket(x, y, updates = 100):
	"""
		Args: 
			x: ndarry, all fatures of X
			y: ndarry, corresponding output label of x
			updates: update iterations on all training data
		Returns:
			the best weighted vector pocket_w.
	"""
	m, n = x.shape
	w = np.zeros(n)
	i = 0
	pocket_w = w
	pocket_error = test_error(x, y, w)
	indexs = np.arange(m)
	while i <= updates:
		# return if no mistake
		errors = indexs[y != hw01_15.sign(x.dot(w))]
		if errors.size == 0:
			break
		# compute a new weight vector w based on a random mistake
		j = random.choice(errors)
		w = w + y[j] * x[j]
		# update my pocket weight vector pocket_w if better than before
		error_cur = test_error(x, y, w)
		if error_cur < pocket_error:
			pocket_w = w
			pocket_error = error_cur
		i += 1
	return pocket_w


def PLA_pocket_naive(x, y, updates=50):
	"""
		Args: 
			x: ndarry, all fatures of X
			y: ndarry, corresponding output label of x
			updates: update iterations on all training data
		Returns:
			the best weighted vector pocket_w.
	"""
	m, n = x.shape
	w = np.zeros(n)
	indexs = np.arange(m)
	for t in range(updates):
		mistakes = indexs[y != hw01_15.sign(x.dot(w))]
		if mistakes.size == 0:
			break
		i = random.choice(mistakes)
		w = w + y[i]*x[i]
	return w


# Q18
def solution_18():
	x_train, y_train = hw01_15.load_data(data_path="hw1_18_train.dat")
	x_test, y_test = hw01_15.load_data(data_path='hw1_18_test.dat')
	epoch = 2000
	all_errors=[]
	for i in range(epoch):
		w = PLA_pocket(x_train, y_train)
		error = test_error(x_test, y_test, w)
		all_errors.append(error)
	print('#Q18: average error is {0}\n'.format(np.mean(all_errors)))
	plt.figure()
	plt.title("Q18-Error Rate versus Frequency Histogram")
	plt.xlabel("Error Value")
	plt.ylabel("Frequency")
	plt.hist(all_errors, rwidth=0.8)
	plt.show()

# Q19
def solution_19():
	x_train, y_train = hw01_15.load_data(data_path="hw1_18_train.dat")
	x_test, y_test = hw01_15.load_data(data_path='hw1_18_test.dat')
	epoch = 200
	all_errors=[]
	for i in range(epoch):
		w = PLA_pocket_naive(x_train, y_train)
		error = test_error(x_test, y_test, w)
		all_errors.append(error)
	print('#Q19: average error is {0}\n'.format(np.mean(all_errors)))
	plt.figure()
	plt.title("Q19-Error Rate versus Frequency Histogram")
	plt.xlabel("Error Value")
	plt.ylabel("Frequency")
	plt.hist(all_errors, rwidth=0.8)
	plt.show()

# Q20
def solution_20():
	x_train, y_train = hw01_15.load_data(data_path="hw1_18_train.dat")
	x_test, y_test = hw01_15.load_data(data_path='hw1_18_test.dat')
	epoch = 200
	all_errors=[]
	for i in range(epoch):
		w = PLA_pocket(x_train, y_train,updates=100)
		error = test_error(x_test, y_test, w)
		all_errors.append(error)
	print('#Q20: average error is {0}\n'.format(np.mean(all_errors)))
	plt.figure()
	plt.title("Q20-Error Rate versus Frequency Histogram")
	plt.xlabel("Error Value")
	plt.ylabel("Frequency")
	plt.hist(all_errors, rwidth=0.8)
	plt.show()


if __name__ == '__main__': 
    
	solution_18()
	# solution_19()
	# solution_20()
