import os
import cv2.cv2
import matplotlib.image as img
'''
这里主要用到了Opencv的人脸检测的分类器haarcascade_frontalface_alt.xml和CascadeClassifier函数
'''
# cascade_path = 'haarcascade_frontalface_alt.xml'

cascade_path = 'D:/anaconda/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml'
cascade = cv2.CascadeClassifier(cascade_path)
images_dir = 'D:/2023PR/face/new_pgmdata'
save_dir = 'D:/2023PR/face/test1'
images = os.listdir(images_dir)
false_num = 0
for image in images:
    img = cv2.imread(os.path.join(images_dir,image),1)
    img_file = images_dir+'/'+image
    # 得到检测到人脸的数量
    # rects = cascade.detectMultiScale(img, 1.3,5)
    # 改变参数
    rects = cascade.detectMultiScale(img, 1.1, 3)
    # print('detected face', len(rects))
    # 如果没有检测到人脸就删除本图片
    if len(rects) == 0:
        cv2.namedWindow('Result',0)
        cv2.imshow('Result', img)
        # cv2.imwrite(save_dir+'/'+str(false_num)+'.jpg',img)
        print(img_file)
        false_num += 1
        os.remove(os.path.join(images_dir, image))
        k = cv2.waitKey(0)
        if k == ord('q'):
            break
print('总共去除%d张图片' % false_num)