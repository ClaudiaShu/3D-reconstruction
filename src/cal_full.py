import numpy as np
import os
import cv2 as cv
import math
import config

##################config########################
data_dir = './data/'#像点和控制点坐标
pic_dir = './pic/'#图片文件
output_dir = './output_dir/'#特征点（fpt)，三维点(3dpt)输出文件

#相机内参矩阵,其中，K[0][0]和K[1][1]代表相机焦距，而K[0][2]和K[1][2]
# f, x0, y0
f = 7935.786962
x0 = 1935.50
y0 = 1295.50
#代表图像的中心像素。
K = np.mat([
        [7935.786962, 0, 1935.5000000000],
        [0, 7935.786962,  1295.5000000000],
        [0, 0, 1.0]])

#选择性删除所选点的范围。
# x = 0.5
# y = 1
MRT = 0.7

w = 3872 
h = 2592

R1 = np.zeros([3, 3])
C_center1=np.mat([0,0,0]).reshape(3,1)
R2 = np.zeros([3, 3])
C_center2=np.mat([0,0,0]).reshape(3,1)

#################################################




#################################################
def cal_a1(phi, omega, kappa):
    a1 = math.cos(phi)*math.cos(kappa)-math.sin(phi)*math.sin(omega)*math.sin(kappa)
    return a1
    # pass

def cal_a2(phi, omega, kappa):
    a2 = -math.cos(phi)*math.sin(kappa)-math.sin(phi)*math.sin(omega)*math.cos(kappa)
    return a2
    # pass

def cal_a3(phi, omega, kappa):
    a3 = -math.sin(phi)*math.cos(omega)
    return a3
    # pass

def cal_b1(phi, omega, kappa):
    b1 = math.cos(omega)*math.sin(kappa)
    return b1
    # pass

def cal_b2(phi, omega, kappa):
    b2 = math.cos(omega)*math.cos(kappa)
    return b2
    # pass

def cal_b3(phi, omega, kappa):
    b3 = -math.sin(omega)
    return b3
    # pass

def cal_c1(phi, omega, kappa):
    c1 = math.sin(phi)*math.cos(kappa)-math.cos(phi)*math.sin(omega)*math.sin(kappa)
    return c1
    # pass

def cal_c2(phi, omega, kappa):
    c2 = -math.sin(phi)*math.sin(kappa)-math.cos(phi)*math.sin(omega)*math.cos(kappa)
    return c2
    # pass

def cal_c3(phi, omega, kappa):
    c3 = math.cos(phi)*math.cos(omega)
    return c3
    # pass


def cal_X(a1, b1, c1, X, Xs, Y, Ys, Z, Zs):
    result = a1*(X-Xs)+b1*(Y-Ys)+c1*(Z-Zs)
    return result
    # pass

def cal_Y(a2, b2, c2, X, Xs, Y, Ys, Z, Zs):
    result = a2*(X-Xs)+b2*(Y-Ys)+c2*(Z-Zs)
    return result
    # pass

def cal_Z(a3, b3, c3, X, Xs, Y, Ys, Z, Zs):
    result = a3*(X-Xs)+b3*(Y-Ys)+c3*(Z-Zs)
    return result
    # pass


def cal_a11(Z_, a1, f, a3, x, x0):
    a11 = (a1*f+a3*x-a3*x0)/Z_
    return a11
    # pass

def cal_a12(Z_, b1, f, b3, x, x0):
    a12 = (b1*f+b3*x-b3*x0)/Z_
    return a12
    # pass

def cal_a13(Z_, c1, f, c3, x, x0):
    a13 = (c1*f+c3*x-c3*x0)/Z_
    return a13
    # pass

def cal_a14(x, x0, y, y0, phi, omega, kappa, f):
    a14 = (y-y0)*math.sin(omega)-((x-x0)*((x-x0)*math.cos(kappa)-(y-y0)*math.sin(kappa))/f+f*math.cos(kappa))*math.cos(omega)
    return a14
    # pass

def cal_a15(x, x0, y, y0, phi, omega, kappa, f):
    a15 = -f*math.sin(kappa)-(x-x0)*((x-x0)*math.sin(kappa)+(y-y0)*math.cos(kappa))/f
    return a15
    # pass

def cal_a16(y, y0):
    a16 = y-y0
    return a16
    # pass

def cal_a21(Z_, a2, f, a3, y, y0):
    a21 = (a2*f+a3*y-a3*y0)/Z_
    return a21
    # pass

def cal_a22(Z_, b2, f, b3, y, y0):
    a22 = (b2*f+b3*y-b3*y0)/Z_
    return a22
    # pass

def cal_a23(Z_, c2, f, c3, y, y0):
    a23 = (c2*f+c3*y-c3*y0)/Z_
    return a23
    # pass

def cal_a24(x, x0, y, y0, phi, omega, kappa, f):
    a24 = -(x-x0)*math.sin(omega)-((y-y0)*((x-x0)*math.cos(kappa)-(y-y0)*math.sin(kappa))/f-f*math.sin(kappa))*math.cos(omega)
    return a24
    # pass

def cal_a25(x, x0, y, y0, phi, omega, kappa, f):
    a25 = -f*math.cos(kappa)-(y-y0)*((x-x0)*math.sin(kappa)+(y-y0)*math.cos(kappa))/f
    return a25
    # pass

def cal_a26(x, x0):
    a26 = -x+x0
    return a26
    # pass

#################################################

def get_coordinate(index):
    file_name = data_dir+"control_point.txt"
    f = open(file_name, 'r')
    for lines in f:
        # print(lines)
        a = int(lines.split(' ')[0])
        X = float(lines.split(' ')[2])
        Y = float(lines.split(' ')[4])
        Z = float(lines.split(' ')[6])
        check_point = int(lines.split(' ')[8])
        #control(1) check(2)
        pts = np.mat([X, Y, Z, 1])
        if(index==a):
            return pts
        else:
            pass

    f.close()

def get_index(name, index):
    file_name = data_dir+name+"_pts_index.txt"
    f = open(file_name, 'r')
    for lines in f:
        # print(lines)
        a = int(lines.split(' ')[0])
        X = int(lines.split(' ')[1])
        Y = int(lines.split(' ')[2])
       
        pts = np.mat([X, Y, 1])
        if(index==a):
            return pts
        else:
            pass

    f.close()

#############DLT###########################

def get_ex(name):
    print(name, '确定DLT所用控制点')
    print("input the number of control points you want to use")
    num = int(input())
    count = 0

    print("input the index of the control point", count+1)
    index_pts_ = int(input())
    obj_pts_ = get_coordinate(index_pts_)
    pts_ = get_index(name, index_pts_)

    coe = np.mat([[-obj_pts_[0,0],-obj_pts_[0,1],-1,0,0,0,pts_[0,0]*obj_pts_[0,0],pts_[0,0]*obj_pts_[0,1]],[0,0,0,-obj_pts_[0,0],-obj_pts_[0,1],-1,pts_[0,1]*obj_pts_[0,0],pts_[0,1]*obj_pts_[0,1]]])
    ini = np.mat([[-pts_[0,0]],[-pts_[0,1]]])
    count = count+1

    while count<num:
        print("input the index of the control point", count+1)
        index_pts = int(input())
        obj_pts = get_coordinate(index_pts)
        pts = get_index(name, index_pts)
        coe_f = np.mat([[-obj_pts[0,0],-obj_pts[0,1],-1,0,0,0,pts[0,0]*obj_pts[0,0],pts[0,0]*obj_pts[0,1]],[0,0,0,-obj_pts[0,0],-obj_pts[0,1],-1,pts[0,1]*obj_pts[0,0],pts[0,1]*obj_pts[0,1]]])
        ini_f = np.mat([[-pts[0,0]],[-pts[0,1]]])

        coe = np.vstack((coe,coe_f))
        ini = np.vstack((ini,ini_f))

        count = count+1


    M = coe.I*ini
    # print(M)

    print(M)#M即为射影变换解算出的八个参数,还需要增加控制点数量
    h1 = M[0,0]
    h2 = M[1,0]
    h3 = M[2,0]
    h4 = M[3,0]
    h5 = M[4,0]
    h6 = M[5,0]
    h7 = M[6,0]
    h8 = M[7,0]

    b1 = h2-h8*x0
    b2 = h5-h8*y0
    b3 = math.sqrt(1/(1+pow(b1/(f*h8),2)+pow(b2/(f*h8),2)))

    a1_a3 = -(h1-h7*x0)/(f*h7)
    a2_a3 = -(h4-h7*y0)/(f*h7)
    

    tan_Kappa = b1/b2
    print(math.atan(tan_Kappa))
    kappa = math.atan(tan_Kappa)

    sin_Omega = -b3
    print(math.asin(-b3))
    omega = math.asin(-b3)

    tan_Phi = 1/(a1_a3*b2-a2_a3*b1)
    print(math.tan(tan_Phi))
    phi = math.tan(tan_Phi)


    X = 0
    Y = 0
    Z = 0

    return phi, omega, kappa, X, Y, Z

#########后方交会##########################

def refine_ex(name):
    phi, omega, kappa, Xs, Ys, Zs = get_ex(name)#得到外参初始值
    
    
    #######输入num个控制点坐标获得X Y Z, obj矩阵为n*4
    print(name,'确定后方交会所用点对')
    print("input the number of control points you want to use")
    num = int(input())
    count = 0

    # print("input the index of the control point", count+1)
    # index_pts_ = int(input())
    # obj_pts = get_coordinate(index_pts_)
    # pts = get_index(name, index_pts_)
    # count = count+1
    while count<num:
        print("input the index of the control point", count+1)
        index_pts = int(input())
        if count == 0:
            obj_pts = get_coordinate(index_pts)
            pts = get_index(name, index_pts)
        elif count > 0:
            obj_pts_ = get_coordinate(index_pts)
            pts_ = get_index(name, index_pts)

            obj_pts = np.vstack((obj_pts, obj_pts_))
            pts = np.vstack((pts, pts_))

        count = count+1

    print(pts)


    # print(obj_pts[0,1])

    ##############迭代计算##########

    not_converge = 1

    while not_converge == 1:
        a1 = cal_a1(phi, omega, kappa)
        a2 = cal_a2(phi, omega, kappa)
        a3 = cal_a3(phi, omega, kappa)
        
        b1 = cal_b1(phi, omega, kappa)
        b2 = cal_b2(phi, omega, kappa)
        b3 = cal_b3(phi, omega, kappa)
        
        c1 = cal_c1(phi, omega, kappa)
        c2 = cal_c2(phi, omega, kappa)
        c3 = cal_c3(phi, omega, kappa)

        i = 0

        
        while i<num:
            X_ = cal_X(a1, b1, c1, obj_pts[i, 0], Xs, obj_pts[i, 1], Ys, obj_pts[i, 2], Zs)
            Y_ = cal_Y(a1, b1, c1, obj_pts[i, 0], Xs, obj_pts[i, 1], Ys, obj_pts[i, 2], Zs)
            Z_ = cal_Z(a1, b1, c1, obj_pts[i, 0], Xs, obj_pts[i, 1], Ys, obj_pts[i, 2], Zs)

            x = pts[i, 0]
            y = pts[i, 1]

            a11 = cal_a11(Z_, a1, f, a3, x, x0)
            a12 = cal_a12(Z_, b1, f, b3, x, x0)
            a13 = cal_a13(Z_, c1, f, c3, x, x0)
            a14 = cal_a14(x, x0, y, y0, phi, omega, kappa, f)
            a15 = cal_a15(x, x0, y, y0, phi, omega, kappa, f)
            a16 = cal_a16(y, y0)

            a21 = cal_a21(Z_, a2, f, a3, y, y0)
            a22 = cal_a22(Z_, b2, f, b3, y, y0)
            a23 = cal_a23(Z_, c2, f, c3, y, y0)
            a24 = cal_a24(x, x0, y, y0, phi, omega, kappa, f)
            a25 = cal_a25(x, x0, y, y0, phi, omega, kappa, f)
            a26 = cal_a26(x, x0)

            x_ = x0-f*X_/Z_
            y_ = y0-f*Y_/Z_

            if i==0:
                A_derivative=np.array([a11, a12, a13, a14, a15, a16, a21, a22, a23, a24, a25, a26]).reshape(2, 6)
                L_mat = np.array([x_, y_]).reshape(2, 1)

            elif  i>0:
                A_derivative_=np.array([a11, a12, a13, a14, a15, a16, a21, a22, a23, a24, a25, a26]).reshape(2, 6)
                L_mat_ = np.array([x_, y_]).reshape(2, 1)

                A_derivative = np.vstack((A_derivative, A_derivative_))
                L_mat = np.vstack((L_mat, L_mat_))

            i = i+1

        print(A_derivative.shape, L_mat.shape)
            
        # ex_mat = (A_derivative.T*A_derivative).I#*A_derivative.T*L_mat
        ATA = np.dot(A_derivative.T, A_derivative)
        ex_mat = np.dot(np.dot(np.linalg.pinv(ATA), A_derivative.T), L_mat)

        # print(ex_mat[1,0])

        Xs = Xs + ex_mat[0,0]
        Ys = Ys + ex_mat[1,0]
        Zs = Zs + ex_mat[2,0]
        phi = phi + ex_mat[3,0]
        omega = omega + ex_mat[4,0]
        kappa = kappa + ex_mat[5,0]

        V_mat = np.dot(A_derivative, ex_mat)-L_mat

        if pow(V_mat[0,0],2)+pow(V_mat[1,0],2)+pow(V_mat[2,0],2)+pow(V_mat[3,0],2)<0.0006:
            not_converge = 0 
            print(V_mat[1,0])


    R = np.zeros([3, 3])
    R[0][0] = cal_a1(phi, omega, kappa)
    R[0][1] = cal_a2(phi, omega, kappa)
    R[0][2] = cal_a3(phi, omega, kappa)

    R[1][0] = cal_b1(phi, omega, kappa)
    R[1][1] = cal_b2(phi, omega, kappa)
    R[1][2] = cal_b3(phi, omega, kappa)

    R[2][0] = cal_c1(phi, omega, kappa)
    R[2][1] = cal_c2(phi, omega, kappa)
    R[2][2] = cal_c3(phi, omega, kappa)

    X_rec = Xs
    Y_rec = Ys
    Z_rec = Zs
    C_center=np.mat([X_rec,Y_rec,Z_rec]).reshape(3,1)

    return R, C_center

#######feature#######

def run_sift(name1, name2):

    img_1 = cv.imread(pic_dir+name1+'.jpg', cv.IMREAD_GRAYSCALE)
    img_2 = cv.imread(pic_dir+name2+'.jpg', cv.IMREAD_GRAYSCALE)

    h1, w1 = img_1.shape
    h2, w2 = img_2.shape

    # SIFT特征计算
    sift = cv.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img_1, None)
    kp2, des2 = sift.detectAndCompute(img_2, None)

    # Flann特征匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)


    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    goodMatch = []

    f1 = open(output_dir+'fpt/'+name1+'_'+name2+'.txt', 'w')
    f2 = open(output_dir+'fpt/'+name2+'_'+name1+'.txt', 'w')
    f1.close
    f2.close

    for m, n in matches:
        # goodMatch是经过筛选的优质配对，如果2个配对中第一匹配的距离小于第二匹配的距离的1/2，基本可以说明这个第一配对是两幅图像中独特的，不重复的特征点,可以保留。
        if m.distance < 0.5*n.distance:
            goodMatch.append(m)

            f1 = open(output_dir+'fpt/'+name1+'_'+name2+'.txt', 'a')
            f2 = open(output_dir+'fpt/'+name2+'_'+name1+'.txt', 'a')
           
            (x1,y1) = kp1[m.queryIdx].pt
            f1.write(str(x1)+'\t'+str(y1)+'\n')

            (x2,y2) = kp2[m.trainIdx].pt
            f2.write(str(x2)+'\t'+str(y2)+'\n')

            f1.close
            f2.close


    # 增加一个维度
    goodMatch = np.expand_dims(goodMatch, 1)
    img_out = cv.drawMatchesKnn(img_1, kp1, img_2, kp2, goodMatch[:15], None, flags=2)

    # cv.namedWindow('image', cv.WINDOW_NORMAL)
    # cv.imshow('image', img_out)#展示图片
    cv.imwrite(output_dir+'sift/'+name2+'_'+name1+'.jpg', img_out)
    print(1)
    # cv.waitKey(0)#等待按键按下
    # cv.destroyAllWindows()#清除所有窗口

########get3d########

def l_mat(In,R,coor):
    l = np.mat(np.zeros((2,3)))
    f = In[0,2]
    xo = In[0,0]
    yo = In[0,1]
    x = coor[0]
    y = coor[1]

    l[0,0] = f*R[0,0] + (x-xo)*R[0,2]
    l[0,1] = f*R[1,0] + (x-xo)*R[1,2]
    l[0,2] = f*R[2,0] + (x-xo)*R[2,2]
    l[1,0] = f*R[0,1] + (y-yo)*R[0,2]
    l[1,1] = f*R[1,1] + (y-yo)*R[1,2]
    l[1,2] = f*R[2,1] + (y-yo)*R[2,2]

    return l
    
def l_approximate(In,R,coor,Ex):
    l_app = np.mat(np.zeros((2,1)))
    f = In[0,2]
    xo = In[0,0]
    yo = In[0,1]
    x = coor[0]
    y = coor[1]
    Xs = Ex[0,0]
    Ys = Ex[1,0]
    Zs = Ex[2,0]

    l_app[0,0] = (f*R[0,0]*Xs + f*R[1,0]*Ys + f*R[2,0]*Zs
         + (x-xo)*R[0,2]*Xs + (x-xo)*R[1,2]*Ys + (x-xo)*R[2,2]*Zs)
    l_app[1,0] = (f*R[0,1]*Xs + f*R[1,1]*Ys + f*R[2,1]*Zs
         + (y-yo)*R[0,2]*Xs + (y-yo)*R[1,2]*Ys + (y-yo)*R[2,2]*Zs)

    return l_app

def get_data(data1, data2):

    data1x = []
    data1y = []
    data2x = []
    data2y = []

    i = 0
    j = 0
    with open(output_dir+'fpt/'+data1+'_'+data2+'.txt', 'r') as f1:
        while True:
            lines = f1.readlines(10000)
            if not lines:
                break
            for line in lines:
                line = line.strip('\n')
                data1x.append(line.split('\t', 1 )[0])
                data1y.append(line.split('\t', 1 )[1])
                i=i+1

        
    with open(output_dir+'fpt/'+data2+'_'+data1+'.txt', 'r') as f2:
        while True:
            lines = f2.readlines(10000)
            if not lines:
                break
            for line in lines:
                line = line.strip('\n')
                data2x.append(line.split('\t', 1 )[0])
                data2y.append(line.split('\t', 1 )[1])
                j=j+1

    if(i!=j):
        print("Wrong matches of keypoints!")
        return 0

    k = 0

    while(k<i):
        run_locate(float(data1x[k]),float(data1y[k]),float(data2x[k]),float(data2y[k]), data1, data2)
        k = k+1   

def run_locate(x1,y1,x2,y2,pic1,pic2):

    #R1, R2分别为左右像片的旋转矩阵，C_center为外方位元素

    #data list
    left_HomonymousImagePoints = (x1, y1) #左同名点 
    right_HomonymousImagePoints = (x2, y2) #右同名点

    #内参（x0,y0,f),相机相同，所以值固定
    #f
    In = np.mat([x0,y0,f])
    K = np.mat([[In[0,2],0,In[0,0]],[0,In[0,2],In[0,1]],[0,0,1]]).reshape(3,3)

    t1 = -R1 * C_center1
    t2 = -R2 * C_center2

    Proj1 = np.mat(K*np.hstack((R1,t1)))
    Proj2 = np.mat(K*np.hstack((R2,t2)))

    calculate_3DX(left_HomonymousImagePoints, right_HomonymousImagePoints, Proj1, Proj2)

def calculate_3DX(kp1, kp2, Proj1, Proj2):
    A0 = np.mat(kp1[0] * Proj1[2,:] - Proj1[0,:])
    A1 = np.mat(kp1[1] * Proj1[2,:] - Proj1[1,:])
    A2 = np.mat(kp2[0] * Proj2[2,:] - Proj2[0,:])
    A3 = np.mat(kp2[1] * Proj2[2,:] - Proj2[1,:])

    train_data = np.mat(np.vstack((A0,A1,A2,A3)))
    U,sigma,VT = np.linalg.svd(train_data)
    posx = VT[3,:].T
    posx_ = posx / posx[3][0]
    position = posx_[0:3]

    # print(position)

    with open(output_dir+'3dpt/'+'cordinate.txt', 'a') as f:
        f.write("%f;%f;%f\n" %(position[0,0],position[1,0],position[2,0]))
        
    return position

########################################

if __name__ == "__main__":

    R1, C_center1 = refine_ex('left')
    R2, C_center2 = refine_ex('right')

    print(0)
    run_sift('left', 'right')
    get_data('left', 'right')

    pass