#!/usr/bin/env python
from pylab import imread, imshow, figure, show, subplot
from numpy import reshape, uint8, flipud
from scipy.cluster.vq import kmeans, vq, whiten
from scipy import misc
import random
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import os
import scipy
import scipy.cluster.hierarchy as sch
import matplotlib.pylab as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.svm import SVC
overlap = 0
n_clusters = 18


def load_data():
	dir_ = '/Users/zpy/Desktop/HMP_Dataset/'
	dics = os.listdir(dir_)
	data_dic = {}

	data = []
	label = []
	label_dic = {}
	ct = 0
	for i in dics:
		if('_MODEL' in i):
			continue
		data_dic[i] = []

		for file_name in os.listdir(dir_+i):
			if(file_name[-3:] != 'txt'):
				continue
			file_path = dir_+i + '/' + file_name
			file = open(file_path, 'r')

			temp_data = []
			for line in file.readlines():
				temp_data.append([int(num) for num in line.strip().split(' ')])
			data_dic[i].append(temp_data)
			data.append(temp_data)
			if(i not in label_dic.keys()):
				label_dic[i] = ct
				ct += 1
			label.append(label_dic[i])
	print(label_dic)
	print(len(label_dic))
	return data_dic, data, label
data_dic, data, label = load_data()

def data_split(data, label, ratio):
	train_size = int(len(label) * ratio)
	train_data = []
	train_label = []
	rem = data
	rem_label = label

	while len(train_data) < train_size:
		index = random.randrange(len(rem))
		a = rem.pop(index)
		b = rem_label.pop(index)
		train_data.append(a)
		train_label.append(b)

	test_data = rem
	test_label= rem_label

	return train_data, train_label, test_data, test_label

def cut_data_to_piece(datas, labels):
	window = 32
	ct = 0
	slices = []
	slices_label = []
	slice_length = []
	for data in datas:
		small_ct = 0
		step = window-overlap
		split_num = (len(data)-window)//step
		rem = len(data) % step
		# print('len(data)', len(data))
		start = window
		for i in range(split_num):

			if(window*i -step>=0):
				#print(start-overlap, min(start -overlap + window, len(data)))
				sli = np.array(data[start -overlap: min(start -overlap + window, len(data))]).reshape(-1)
				start = start -overlap + window
			else:
				sli = np.array(data[window * i : window * (i + 1)]).reshape(-1)
			slices.append(sli)
			slices_label.append(labels[ct])
			small_ct += 1
		if(rem != 0):
			slices.append(np.array(data[-window:]).reshape(-1))
			slices_label.append(labels[ct])
			small_ct += 1
		slice_length.append(small_ct) #length
		ct += 1
	return slices, slices_label, slice_length

def plot_hist(centers, i):
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots()
	plt.bar(np.arange(20), centers/np.sum(centers))

	#plt.grid(axis='y', alpha=0.75)
	plt.xlabel('cluster')
	my_x_ticks = np.arange(0, 20, 1)
	plt.xticks(my_x_ticks)
	plt.ylabel('Probability')
	plt.title('Histogram for the cluster centers %d'%i)
	plt.text(23, 45, r'$\mu=15, b=3$')
	plt.savefig('cluster_'+str(i)+'.eps')
	# plt.show()

def cluster(n_clusters, slices, te_slices):
	#clusters = AgglomerativeClustering(linkage='average', n_clusters=n_clusters)  # Z.shape
	clusters = KMeans(n_clusters=n_clusters, max_iter=500)
	clusters.fit(slices)
	train_cluster_labels = clusters.predict(slices)
	test_cluster_labels = clusters.predict(te_slices)
	return train_cluster_labels, test_cluster_labels#cluster_labels

def make_hists(n_clusters, cluster_labels, slice_length):
	start_pos = 0
	new_features = []
	bins = [i for i in range(n_clusters + 1)]

	for i in slice_length:
		cluster_result = cluster_labels[start_pos: start_pos + i]
		start_pos += i
		hist, _ = np.histogram(cluster_result, bins=bins)  # , density=True)
		new_features.append(hist)

	return new_features

def VQ(slices, slices_label, slice_length, te_slices, te_slice_length):
	# first layer
	cluster_labels, test_cluster_labels = cluster(n_clusters, slices, te_slices)
	train_features = make_hists(n_clusters, cluster_labels, slice_length)
	test_features = make_hists(n_clusters, test_cluster_labels, te_slice_length)
	return train_features, test_features # new_features

def norm(features):
	f = np.array(features)
	shape = f.shape
	for i in range(shape[0]):
		#max_.append(np.maximun(f[:, i]))
		#min_.append(np.minimum(f[:, i]))
		max_ = np.max(f[i, :])
		min_ = np.min(f[i, :])
		f[i, :] = (f[i, :] - min_)/(max_ - min_)
		# print(f[:, i])
	return f
def norm2(features):
	f = np.array(features)
	shape = f.shape
	for i in range(shape[1]):
		#max_.append(np.maximun(f[:, i]))
		#min_.append(np.minimum(f[:, i]))
		max_ = np.max(f[:,i])
		min_ = np.min(f[:,i])
		f[:,i] = (f[:,i] - min_)/(max_ - min_)
		# print(f[:, i])
	return f
def classify(X, Y, test_x, test_y):
	clf = SVC(kernel='linear')
	clf.fit(X, Y) #norm(
	preds = clf.predict(test_x) #norm()
	acc = accuracy_score(preds, test_y)
	print(acc)
	C = confusion_matrix(preds, test_y)
	print(C.shape)
	print(C)

train_data, train_label, test_data, test_label = data_split(data, label, 0.8)
print('train data & test data: ', len(train_label), len(test_data))
tr_slices, tr_slices_label, tr_slice_length = cut_data_to_piece(train_data, train_label)
te_slices, te_slices_label, te_slice_length = cut_data_to_piece(test_data, test_label)
train_features, test_features = VQ(tr_slices, tr_slices_label, tr_slice_length, te_slices, te_slice_length)

for i in range(3):
	classify(train_features, train_label, test_features, test_label)

def hist():
	hist_class = {}
	for i in range(len(train_features)):
		if (train_label[i] not in hist_class.keys()):
			hist_class[train_label[i]] = []
		hist_class[train_label[i]].append(train_features[i])
	for i in hist_class.keys():
		means_ = np.mean(np.array(hist_class[i]), axis=0)
		plot_hist(means_, i)
