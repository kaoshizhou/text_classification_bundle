with open('..//reference_data//abalone_test.solution', 'r') as f:
    text = f.read()
    lst = text.rstrip('\n').split('\n')
    for i in range(len(lst)):
        if int(lst[i]) < 10:
            lst[i] = '1 0'
        else:
            lst[i] = '0 1'
        # if lst[i] == '3':
        #     lst[i] = '0 0 1 0 0 0'
        # if lst[i] == '5':
        #     lst[i] = '0 0 0 1 0 0'
        # if lst[i] == '6':
        #     lst[i] = '0 0 0 0 1 0'
        # if lst[i] == '7':
        #     lst[i] = '0 0 0 0 0 1'

str = '\n'.join(lst)
with open('..//reference_data//abalone_test.solution', 'w') as f:
    f.write(str)