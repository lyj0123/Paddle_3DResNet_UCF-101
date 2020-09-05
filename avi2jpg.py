import os
import numpy as np
import cv2

video_src_src_path = 'data/UCF-101'         # 视频数据所在源文件根目录
video_target_path = 'lyj/UCF-101_jpg'      # 视频分解后存储的目标路径
if not os.path.exists(video_target_path):
    os.mkdir(video_target_path)
label_name = os.listdir(video_src_src_path) # 列出每个类的文件夹名
label_dir = {}  ## 存储每个视频生成的图片
index = 0       ## 索引
for i in label_name:    ## i 指向具体的某类文件夹
    if i.startswith('.'):
        continue
    label_dir[i] = index

    print('avi2jpg.py 中 ，  label_name ', i , ' index ', index )

    index += 1
    video_src_path = os.path.join(video_src_src_path, i)
    video_save_path = os.path.join(video_target_path, i) + '_jpg'
    if not os.path.exists(video_save_path):
        os.mkdir(video_save_path)

    videos = os.listdir(video_src_path)     # 一类文件夹中的视频列表
    # 过滤出avi文件，只对 mp4 文件进行分解
    videos = filter(lambda x: x.endswith('avi'), videos)

    for each_video in videos:
        each_video_name, _ = each_video.split('.')
        if not os.path.exists(video_save_path + '/' + each_video_name):
            os.mkdir(video_save_path + '/' + each_video_name)
        # 存储一个视频生成图片的文件夹        这一类视频文件夹名（_jpg） + 视频名 + /
        each_video_save_full_path = os.path.join(video_save_path, each_video_name) + '/'
        # 视频所在的原路径
        each_video_full_path = os.path.join(video_src_path, each_video)
        ## VideoCapture是用于从视频文件、图片序列、摄像头捕获视频的类
        cap = cv2.VideoCapture(each_video_full_path)
        frame_count = 1
        success = True
        while success:
            success, frame = cap.read()
            # cap.read()按帧读取视频，ret (success),frame是获cap.read()方法的两个返回值。
            # 其中 ret (success) 是布尔值，如果读取帧是正确的则返回True，
            # 如果文件读取到结尾，它的返回值就为False。
            # frame就是每一帧的图像，是个三维矩阵。
            # print('read a new frame:', success)

            params = []
            params.append(1)## 特定格式保存的参数编码，默认值std::vector<int>()  一般可以不写
            if success:
                # cv2.imwrite(each_video_save_full_path + each_video_name + "_%d.jpg" % frame_count, frame, params)
                # 作业中抽帧的视频质量比较高，改成下面的语句之后，实现百分之75的品质
                cv2.imwrite(each_video_save_full_path + each_video_name + "_%d.jpg" % frame_count, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            frame_count += 1
        cap.release()
np.save('label_dir.npy', label_dir)
print(label_dir)
print("源文件下有类别数：",len(label_name))
