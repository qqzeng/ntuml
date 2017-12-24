#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
=====
@author: qqzeng
@email: qtozeng@gmail.com
@date: 2017/12/24
=====

@desc: the program of the 15th, 16th and 17th in hw#1 of ntuml
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
"""
PLA 本身比较简单，关键需要理解为什么采取那种更新方式。另外，其收敛性也是需要考虑的。
"""
def load_data(data_path="hw1_15_train.dat"):
    """
        Args:
            data_path: the training data path
        Returns:
            a array of training data
    """
    my_data_arr = np.loadtxt(data_path)
    m, n = my_data_arr.shape
    # remeber x0 = 1
    x = np.hstack((np.ones((m, 1)), my_data_arr[:,:-1]))
    y = my_data_arr[:,-1]
    return x, y

def sign(x):
	"""
		Args: 
			x: input number or array
		Returns:
			the answer array of operations on input
	"""
	vfunc = np.vectorize(lambda t: 1 if t > 0 else -1)
	return vfunc(x)

def PLA_naive(x, y, alpha = 1.0):
	"""
		Args: 
			x: ndarry, all fatures of X
			y: ndarry, corresponding output label of x
			alpha: frequent size of learning rate
		Returns:
			some statistics about PLA.
	"""
	m, n = x.shape
	w = np.zeros(n)
	frequent = 0
	i = 0
	ok_count = 0
	epsilon = 0.0001
	error_index = -1
	# other halt conditions will be ok
	while (abs(ok_count - m) / m) >= epsilon:
		if y[i] != sign(x[i].dot(w)):
			w += alpha * y[i] * x[i]
			frequent += 1
			error_index = i
			ok_count = 0
		else:
			ok_count += 1 # reset ok_count
		i += 1
		if i >= m - 1:# continue to cycle
			i = 0
	return w, frequent, error_index


def PLA_random(x, y, alpha = 1.0):
	"""
		Args: 
			x: ndarry, all fatures of X
			y: ndarry, corresponding output label of x
			alpha: frequent size of learning rate
		Returns:
			some statistics about PLA.
	"""
	m, n = x.shape
	w = np.zeros(n)
	frequent = 0
	i = 0
	ok_count = 0
	epsilon = 0.0001
	error_index = -1
	indexs_rand = np.arange(m)
	random.shuffle(indexs_rand)
	j = -1
	# other halt conditions will be ok
	while (abs(ok_count - m) / m) >= epsilon:
		j = indexs_rand[i]
		if y[j] != sign(x[j].dot(w)):
			w += alpha * y[j] * x[j]
			frequent += 1
			error_index = i
			ok_count = 0
		else:
			ok_count += 1 # reset ok_count
		i += 1
		if i >= m - 1:# continue to cycle
			i = 0
	return w, frequent, error_index

# Q15
def solution_15():
    x, y = load_data(data_path="hw1_15_train.dat")
    _, frequent, error_index = PLA_naive(x, y)
    print('#Q15: number of updates is {0} and the last error index is {1}\n'
    	.format(frequent, error_index))

# Q16
def solution_16():
	x, y = load_data(data_path="hw1_15_train.dat")
	epoch = 2000
	frequents = [PLA_random(x, y)[1] for i in range(epoch)]
	print('#Q16: average number of updates is {0}\n'.format(np.mean(frequents)))
	# figure Histogram
    # for details see https://matplotlib.org/2.0.2/examples/statistics/index.html
	plt.figure()
	plt.title("Q16-Number of Updates versus Frequency Histogram")
	plt.xlabel("Updates Value")
	plt.ylabel("Frequency")
	plt.hist(frequents, rwidth=0.8)
	plt.show()

# Q17
def solution_17():
	x, y = load_data(data_path="hw1_15_train.dat")
	epoch = 2000
	frequents = [PLA_random(x, y, alpha = 0.5)[1] for i in range(epoch)]
	print('#Q17: average number of updates is {0}\n'.format(np.mean(frequents)))
	plt.figure()
	plt.title("Q17-Number of Updates versus Frequency Histogram")
	plt.xlabel("Updates Value")
	plt.ylabel("Frequency")
	plt.hist(frequents, rwidth=0.8)
	plt.show()

def solution_17_2():
	x, y = load_data(data_path="hw1_15_train.dat")
	epoch = 2000
	# frequents_16 = [PLA_random(x, y)[1] for i in range(epoch)]
	# frequents_17 = [PLA_random(x, y, alpha = 0.5)[1] for i in range(epoch)]
	# histogram=plt.figure()
	# plt.hist(frequents_16, alpha=0.5)
	# plt.hist(frequents_17, alpha=0.5)
	# plt.show()

	# frequents_16 = [PLA_random(x, y)[1] for i in range(epoch)]
	# frequents_17 = [PLA_random(x, y, alpha = 0.5)[1] for i in range(epoch)]
	# frequents_all = [frequents_16, frequents_17]
	# plt.figure()
	# plt.title("Q16 VS Q17 on Number of Updates")
	# plt.xlabel("Value")
	# plt.ylabel("Frequency")
	# plt.hist(frequents_all, histtype='bar')
	# plt.show()

	frequents_16 = [PLA_random(x, y)[1] for i in range(epoch)]
	frequents_17 = [PLA_random(x, y, alpha = 0.5)[1] for i in range(epoch)]
	frequents_all = np.vstack((np.asarray(frequents_16), np.asarray(frequents_17))).T
	plt.figure()
	blue_patch = mpatches.Patch(color='blue', label='frequents_17')
	orange_patch = mpatches.Patch(color='orange', label='frequents_16')
	plt.legend(handles=[blue_patch, orange_patch])
	plt.title("Q16 VS Q17 on Number of Updates")
	plt.xlabel("Value")
	plt.ylabel("Frequency")
	plt.hist(frequents_all, normed=1, histtype='bar')
	plt.show()


if __name__ == '__main__': 
	# solution_15()
    # solution_16()
    # solution_17()
    solution_17_2();