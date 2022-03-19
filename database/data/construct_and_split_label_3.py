import sys
import os
import random

if __name__ == '__main__':

    # 数据集名称
    image_folder = 'image'
    label_folder = 'label'
    edge_folder = 'edge'

    # 标签文件名称
    all_sample_txt = 'label_01.txt'
    test_txt = 'label_01_test.txt'
    train_txt = 'label_01_train.txt'

    # 训练集比例
    train_ratio = 0.9

    # 打标签模式
    random_mode = 'all'
    if random_mode == 'file':
        all_random = False
        file_random = True
    elif random_mode == 'all':
        all_random = True
        file_random = False
    else:
        print('Error! Please set random_mode file or all')
        sys.exit()

    input_folderList = os.listdir(image_folder)
    label_imgList = os.listdir(label_folder)

    print(f'the number of labels is: {len(label_imgList)}.')

    sample_all_dict = {}

    for folder in input_folderList:
        sample_list = []
        img_list = os.listdir(os.path.join(image_folder, folder))
        print(f'the number of {folder} is: {len(img_list)}.')

        for imgs_fn in img_list:
            if imgs_fn in label_imgList:
                sample_list.append('{} {} {}\n'.format(os.path.join(image_folder, folder, imgs_fn),
                                                    os.path.join(label_folder, imgs_fn),
                                                    os.path.join(edge_folder, imgs_fn)))
            else:
                print(f'Error! The {imgs_fn} in folder: {folder}'
                      f' cannot be found in folder: {label_folder}.')
                sys.exit()

        sample_all_dict[f'{folder}'] = sample_list

    with open(all_sample_txt, 'w') as f:
        for _, sample_list in sample_all_dict.items():
            f.writelines(sample_list)

    print(f'{all_sample_txt}, have been constructed!')

    train_list = []
    test_list = []

    sapmle_len = len(list(sample_all_dict.values())[0])
    all_sapmle_len = len(list(range(sapmle_len * len(list(sample_all_dict.values())))))
    index_list = []

    # 全部随机
    if all_random == True:
        index_list = list(range(all_sapmle_len))
        random.shuffle(index_list)
        all_sample_list = []

        for i in range(len(list(sample_all_dict.values()))):
            all_sample_list.extend(list(sample_all_dict.values())[i])

        train_index = index_list[:int(all_sapmle_len * train_ratio)]
        test_index = index_list[int(all_sapmle_len * train_ratio):]

        train_list.extend([all_sample_list[idx] for idx in train_index])
        test_list.extend([all_sample_list[idx] for idx in test_index])

        print(f'train:{len(train_index)}, test:{len(test_index)}.')

    # 部分随机
    if file_random == True:
        for folder_name, sample_list in sample_all_dict.items():
            index_list = list(range(sapmle_len))
            random.shuffle(index_list)

            train_index = index_list[:int(sapmle_len * train_ratio)]
            test_index = index_list[int(sapmle_len * train_ratio):]

            train_list.extend([sample_list[idx] for idx in train_index])
            test_list.extend([sample_list[idx] for idx in test_index])

            print(f'{folder_name}, train:{len(train_index)}, test:{len(test_index)}.')

    with open(train_txt, 'w') as f:
        f.writelines(train_list)

    with open(test_txt, 'w') as f:
        f.writelines(test_list)

    print(f'{train_txt} and {test_txt}, have been constructed!')
