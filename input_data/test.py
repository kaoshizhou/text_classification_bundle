# this file is used for dataset preprocess, you can preprocess the data
# on your own way, and this file is just an example.
#
# *.origin ---> *_train/valid/test.data/solution

import glob
import numpy as np

# you should save the original dataset with the .origin extension
origin_file = 'glass.origin'

train_data_file = 'glass_train.data'
train_solution_file = 'glass_train.solution'
valid_data_file = 'glass_valid.data'
valid_solution_file = '../reference_data/glass_valid.solution'
test_data_file = 'glass_test.data'
test_solution_file = '../reference_data/glass_test.solution'
number = 55000

shuffle_data = np.arange(number)
np.random.shuffle(shuffle_data)

# data = np.loadtxt('O4_train.data', dtype=np.str, delimiter='\n', encoding='utf-8')
# label = np.loadtxt('O4_train.solution', dtype=np.str, delimiter='\n', encoding='utf-8')

with open('O4_train.data', 'r', encoding='utf-8') as f:
    data = f.read()
    data = data.rstrip('\n').split('\n')
    data = np.array(data, dtype=np.str)

with open('O4_train.solution', 'r', encoding='utf-8') as f:
    label = f.read()
    label = label.rstrip('\n').split('\n')
    label = np.array(label, dtype=np.str)

data1 = []
label1 = []

for i in range(len(shuffle_data)):
    data1.append(data[shuffle_data[i]])
    label1.append(label[shuffle_data[i]])

del(data)
del(label)
# data = data[shuffle_data]
# label = label[shuffle_data]

data1 = np.array(data1, dtype=np.str)
label1 = np.array(label1, dtype=np.str)

train_num = int(number * 0.8)
valid_num = number - train_num

train_data_set = data1[0:train_num]
train_label_set = label1[0:train_num]
valid_data_set = data1[-valid_num:]
valid_label_set = label1[-valid_num:]

np.savetxt('O4_train.data1', train_data_set, fmt='%s', encoding='utf-8')
np.savetxt('O4_train.solution1', train_label_set, fmt='%s', encoding='utf-8')
np.savetxt('O4_valid.data1', valid_data_set, fmt='%s', encoding='utf-8')
np.savetxt('O4_valid.solution1', valid_label_set, fmt='%s', encoding='utf-8')
# total_num = data.shape[0]
# # the dataset is divided as 0.6 0.1 0.3 of total_number
# train_num = int(total_num * 0.6)
# valid_num = int(total_num * 0.1)
# test_num = total_num - train_num - valid_num

# train_set = data[0:train_num]
# valid_set = data[train_num:train_num+valid_num]
# test_set = data[-test_num:]

# np.savetxt(train_data_file, train_set[:,1:-1], delimiter=' ', fmt='%s')
# np.savetxt(train_solution_file, train_set[:,-1], delimiter=' ', fmt='%s')
# np.savetxt(valid_data_file, valid_set[:,1:-1], delimiter=' ', fmt='%s')
# np.savetxt(valid_solution_file, valid_set[:,-1], delimiter=' ', fmt='%s')
# np.savetxt(test_data_file, test_set[:,1:-1], delimiter=' ', fmt='%s')
# np.savetxt(test_solution_file, test_set[:,-1], delimiter=' ', fmt='%s')

