import imutils
import cv2
import numpy as np
import os


def find_center(img_list):

    for img in img_list:
        # 转到HSV空间
        image = cv2.imread(img)
        hue_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 用颜色分割图像 取出指定颜色的像素
        low_range = np.array([0, 0, 0])
        high_range = np.array([10, 10, 10])

        # inRange得到的是二值图像 在范围内的像素置为白色 范围之外的设为黑色
        th = cv2.inRange(hue_image, low_range, high_range)

        # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        # cv2.imshow('img', th)
        # cv2.waitKey(0)

        # 高斯平滑轮廓
        blur = cv2.GaussianBlur(th, (3, 3), 0)

        # 膨胀
        dilated = cv2.dilate(blur, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        # can = cv2.Canny(blur, 200, 255)

        # findContours 找到外部轮廓  该函数只接受二值图像即黑白的 灰度图也不行
        cnts = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cnts = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # cnts = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # 用以区分OpenCV2.4和OpenCV3
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        print('center number:', len(cnts))
        # f = open('./data/right_pts.txt', 'w')
        f = open('./data/left_pts.txt', 'w')

        for c in cnts:
            # 获取中心点
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            print(cX, cY)
            cv2.putText(image, '('+str(cX)+','+str(cY)+')', (cX+10, cY+10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 0), 3)
            #cv2.putText(图像, 文字, (x, y), 字体, 大小, (b, g, r), 宽度)
            f.write(str(cX)+' '+str(cY)+'\n')


            # 画出轮廓和中点
            # cv2.drawContours(image, [c], -1, (255, 255, 0), 1)
            cv2.circle(image, (cX, cY), 1, (0, 255, 0), -1)
            # cv2.putText(image, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # 显示图像
        cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        cv2.imshow("Image", image)
        cv2.imwrite("./data/left_center.jpg", image)
        cv2.waitKey(3000)

    cv2.destroyAllWindows()

base_dir = './data'
img_list = [os.path.join(base_dir, filename) for filename in os.listdir(base_dir)]
# img_list = ['./data/0.png', './data/1.png', './data/3.png', './data/4.png']
# img_list = ['./data/3.png']
# img_list = ['./data/left.jpg','./data/right.jpg']
img_list = ['./data/left.jpg']
find_center(img_list)

