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

data = np.loadtxt(origin_file, dtype=np.str, delimiter=',')
np.random.shuffle(data)

total_num = data.shape[0]
# the dataset is divided as 0.6 0.1 0.3 of total_number
train_num = int(total_num * 0.6)
valid_num = int(total_num * 0.1)
test_num = total_num - train_num - valid_num

train_set = data[0:train_num]
valid_set = data[train_num:train_num+valid_num]
test_set = data[-test_num:]

np.savetxt(train_data_file, train_set[:,1:-1], delimiter=' ', fmt='%s')
np.savetxt(train_solution_file, train_set[:,-1], delimiter=' ', fmt='%s')
np.savetxt(valid_data_file, valid_set[:,1:-1], delimiter=' ', fmt='%s')
np.savetxt(valid_solution_file, valid_set[:,-1], delimiter=' ', fmt='%s')
np.savetxt(test_data_file, test_set[:,1:-1], delimiter=' ', fmt='%s')
np.savetxt(test_solution_file, test_set[:,-1], delimiter=' ', fmt='%s')

