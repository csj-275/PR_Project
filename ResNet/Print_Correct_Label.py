# target_element = 1223

def find_element_2d(target, matrix):
    for i, row in enumerate(matrix):
        for j, element in enumerate(row):
            if element == target:
                return i, j
    return -1, -1  # 返回 -1, -1 表示没有找到目标元素


def Print_Correct_Label(target_element, Class, print_flag):
    ID = []
    with open('D:/python_pro/faceDR', 'r') as f:
        for line in f.readlines():
            a = line[1:].split(' (', 6)
            if int(a[0]) in [1228, 1232, 1808, 4056, 4135, 4136, 5004]:
                continue
            ID.append(a)
    with open('D:/python_pro/faceDS', 'r') as f:
        for line in f.readlines():
            a = line[1:].split(' (', 6)
            if int(a[0]) in [1228, 1232, 1808, 4056, 4135, 4136, 5004]:
                continue
            ID.append(a)

    row_index, col_index = find_element_2d(str(target_element), ID)

    # print(ID, type(ID))

    # if row_index == -1:
    #     print("目标元素未找到")
    # else:
    #     print(f"目标元素 {target_element} 的位置为：行 {row_index}，列 {col_index}")
    if Class == 'SEX':
        if print_flag:
            print(ID[row_index][1][6:-1])
        return ID[row_index][1][6:-1]
    elif Class == 'AGE':
        if print_flag:
            print(ID[row_index][2][6:-1])
        return ID[row_index][2][6:-1]
    elif Class == 'RACE':
        if print_flag:
            print(ID[row_index][3][6:-1])
        return ID[row_index][3][6:-1]
    elif Class == 'FACE':
        if print_flag:
            print(ID[row_index][4][6:-1])
        return ID[row_index][4][6:-1]
    elif Class == 'PROP':
        if print_flag:
            print('none' if ID[row_index][5][8:-4] == '' else ID[row_index][5][8:-4])
        return 'none' if ID[row_index][5][8:-4] == '' else ID[row_index][5][8:-4]