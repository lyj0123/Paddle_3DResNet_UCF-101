import os
import numpy as np
import pickle


label_dic = np.load('label_dir.npy', allow_pickle=True).item()
print("\n标签字典 (label_dic) ：\n",label_dic)    # 输出字典

source_dir = 'lyj/UCF-101_jpg'
target_train_dir = 'lyj/train'
target_test_dir = 'lyj/test'
if not os.path.exists(target_train_dir):
    os.mkdir(target_train_dir)
if not os.path.exists(target_test_dir):
    os.mkdir(target_test_dir)


### 目的：根据 trainlist01.txt 和 testlist01.txt 文件，分出哪些视频应该属于训练集，哪些应该属于测试集
trainlist = 'lyj/trainlist01.txt'   # 'trainlist01.txt' 中存储内容为：类名/视频名  所对应的数字类别
f = open( trainlist , "r" )
train_lines = f.readlines()
train_video_name  = []
for line in train_lines:
    train_name  = line.strip().split()[0].split('/')[1].split('.')[0]
    train_video_name.append(train_name)
f.close()

testlist  = 'lyj/testlist01.txt'    # 'testlist01.txt'  中存储内容为：类名/视频名
f = open(testlist, 'r')
test_lines = f.readlines()
test_video_name = []
for line in test_lines:
    test_name = line.strip().split('/')[1].split('.')[0]
    test_video_name.append(test_name)
f.close()


for key in label_dic:       # key 指向的是每一类视频的根目录名称
    each_mulu = key + '_jpg'
    # print(each_mulu, key) # 某一类视频的类名_jpg   某一类视频的类名
    label_dir = os.path.join(source_dir, each_mulu) # 某一类视频，分解后的根目录
    label_mulu = os.listdir(label_dir)  # 某一类视频文件夹中的所有视频名称列表

    for each_label_mulu in label_mulu:  # label_mulu 某一类视频中的所有视频名称, each_label_mulu 具体的一个视频
        image_file = os.listdir(os.path.join(label_dir, each_label_mulu))   # image_file 某一个视频解压后的所有图片名称
        image_file.sort()               # 根据图片名称进行排序
        # -6 原因： '_1.jpg'(_id.jpg) 占了最后六个字符，id是数字 ；  [0] 表示第一张图片，任意不超范围的数字都可
        # image_name = image_file[0][:-6]   # image_name 与 each_label_mulu 是相同的 表示的是视频名称(该图片从哪个视频中分解出来的)
        image_name = each_label_mulu
        image_num = len(image_file)     # 表示该视频分解后，有多少张图片
        frame = []                      # frame 存储一个视频分解后，所有图片的绝对路径
        vid = image_name
        for i in range(image_num):
            # i+1 视频分解后下标从1开始
            # image_path = os.path.join(os.path.join(label_dir, each_label_mulu), image_name + '_' + str(i+1) + '.jpg')
            # label_dir 视频类名(属于哪一类)；each_label_mulu 视频名(具体的某一个视频名称)  image_file[i] 图片名称
            image_path = os.path.join(os.path.join(label_dir, each_label_mulu), image_file[i])  # image_path 表示一张图片的绝对路径
            frame.append(image_path)


        output_pkl = vid + '.pkl'
        ### 根据 vid(视频名称) 判断视频是  属于训练集  还是  属于测试集
        if vid in train_video_name:
            output_pkl = os.path.join(target_train_dir, output_pkl)
        elif vid in test_video_name:
            output_pkl = os.path.join(target_test_dir, output_pkl)

        f = open(output_pkl, 'wb')
        ## -1： 当参数 protocal 的值是负数， 使用最高 protocal 对 obj 压缩。
        pickle.dump((vid, label_dic[key], frame), f, -1)
        f.close()

print('\n\n训练集 和 测试集的 pkl 文件生成成功')
    



