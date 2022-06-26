import sys
import os
import random


def check_file_ext(fileList,obj_ext):
    for filename in fileList:
        fn, ext = os.path.splitext(filename)
        if ext != obj_ext:
            return False
    return True


if __name__ == '__main__':
    # 首先，image里面子目录按照例子建立，label图像放置也按照例子。

    # 保证label图像名与image子目录中的图像名一一对应

    # 其次，image_folder label_folder填入文件夹名

    # all_sample_txt 为所对应的标签文件，并且会产生对应的test和train文件

    # 最后，train_ratio为训练集样本比例，ins_random 表示每个子目录下随机训练集和测试集
    # folder_random 表示所有子目录统一随机
    # 当ins_random = True，则folder_random选项无效

    image_folder = 'image'  #
    label_folder = 'label'
    all_sample_txt = 'label_01.txt'

    train_ratio = 0.92

    ins_random = True
    folder_random = True

    folder_random = False if ins_random == True else folder_random


    input1_folderList = os.listdir(image_folder)
    label_imgList = os.listdir(label_folder)

    print(f'the number of labels is: {len(label_imgList)}.')
    
    # sample_all_list = []
    sample_all_dict = {}
    for folder in input1_folderList:
        sample_list = []
        img_list = os.listdir(os.path.join(image_folder,folder))
        print(f'the number of {folder} is: {len(img_list)}.')

        for imgs_fn in img_list:
            fn_index = imgs_fn.find('_', 3) + 1
            # print(fn_index)
            imgs_fn_short = imgs_fn[fn_index:]
            if imgs_fn_short in label_imgList:
                sample_list.append('{} {}\n'.format(os.path.join(image_folder, folder, imgs_fn),
                                                    os.path.join(label_folder, imgs_fn_short)))
            else:
                print(f'Error! The {imgs_fn} in folder: {folder}'
                  f' cannot be found in folder: {label_folder}.')
                sys.exit()
    
         # sample_all_list.append(sample_list)
        sample_all_dict[f'{folder}'] = sample_list

    with open(all_sample_txt, 'w') as f:
        for _, sample_list in sample_all_dict.items():
            f.writelines(sample_list)

    #####################################################################

    fn, ext = os.path.splitext(all_sample_txt)
    train_txt = f'{fn}_train{ext}'
    test_txt = f'{fn}_test{ext}'

    train_list = []
    test_list = []

    sapmle_len = len(list(sample_all_dict.values())[0])

    index_list = list(range(sapmle_len))
    if folder_random == True:
        random.shuffle(index_list)

    for folder_name, sample_list in sample_all_dict.items():
        if ins_random == True:
            index_list = list(range(sapmle_len))
            random.shuffle(index_list)

        train_index = index_list[:int(sapmle_len*train_ratio)]
        test_index = index_list[int(sapmle_len*train_ratio):]

        train_list.extend([sample_list[idx] for idx in train_index])
        test_list.extend([sample_list[idx] for idx in test_index])

        print(f'{folder_name},train_test:{len(train_index)},{len(test_index)}')

    with open(train_txt, 'w') as f:
        f.writelines(train_list)

    with open(test_txt, 'w') as f:
        f.writelines(test_list)

    print(f'{train_txt},{train_txt},{all_sample_txt} have been constructed!')
