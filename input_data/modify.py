import glob
import os
import re

public_file = glob.glob('glass_public.info')
train_data_file = glob.glob('glass_train.data')
train_solution_file = glob.glob('glass_train.solution')
valid_data_file = glob.glob('glass_valid.data')
test_data_file = glob.glob('glass_test.data')

with open(train_data_file[0], 'r', encoding='utf-8') as f:
    data = f.read()
    train_num = len(data.rstrip('\n').split('\n'))
    feat_num = len(data.split('\n')[0].split(' '))


with open(train_solution_file[0], 'r', encoding='utf-8') as f:
    data = f.read()
    label_num = len(data.split('\n')[0].split(' '))

with open(valid_data_file[0], 'r', encoding='utf-8') as f:
    data = f.read()
    valid_num = len(data.rstrip('\n').split('\n'))

with open(test_data_file[0], 'r', encoding='utf-8') as f:
    data = f.read()
    test_num = len(data.rstrip('\n').split('\n'))

with open(public_file[0], 'r', encoding='utf-8') as fp:
    text = fp.read()
    text = text.replace(re.findall(r'feat_num = .*', text)[0], 'feat_num = '+str(feat_num))
    text = text.replace(re.findall(r'label_num = .*', text)[0], 'label_num = '+str(label_num))
    text = text.replace(re.findall(r'train_num = .*', text)[0], 'train_num = '+str(train_num))
    text = text.replace(re.findall(r'valid_num = .*', text)[0], 'valid_num = '+str(valid_num))
    text = text.replace(re.findall(r'test_num = .*', text)[0], 'test_num = '+str(test_num))

with open(public_file[0], 'w', encoding='utf-8') as fp:
    fp.write(text)
    