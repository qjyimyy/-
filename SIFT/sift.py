import cv2
import numpy as np
from math import sin, cos, pi



def Sift(img):

    r, c = np.shape(img)
    # 提取角点
    corners = [[int(i[0][0]), int(i[0][1])]
               for i in cv2.goodFeaturesToTrack(img, 233, 0.01, 10)]
    img = cv2.GaussianBlur(img, (5, 5), 1, 1)
    img = np.array(img, dtype="float")


    kernel = np.array([
        [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
        [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]], dtype="float") / 6
    gx = cv2.filter2D(img, -1, np.array(kernel[1]))
    gy = cv2.filter2D(img, -1, np.array(kernel[0]))
    # 返回梯度以及方向
    gradient, angle = cv2.cartToPolar(gx, gy)
    # 角点总数为length，邻域为bins
    bins = (r + c) // 80
    length = len(corners)

    def mainDir():
        direct = [] # 存储每个角点的主方向
        for corner in corners:
            y, x = corner
            voting = [0 for i in range(37)]
            # voting下标是各个bin，值为bin的得票数
            for i in range(max(x-bins, 0), min(x+bins+1, r)):
                for j in range(max(y-bins, 0), min(y+bins+1, c)):
                    k = int((angle[i][j]+pi) / (pi/18) + 1)
                    if k >= 37:
                        k = 36
                    # 用梯度幅值作为权重投票
                    voting[k] += gradient[i][j]
            # 找出最大的
            d = 1
            for i in range(2, 37):
                if voting[i] > voting[d]:
                    d = i
            # 转化成弧度
            direct.append((d / 18 - 1 - 1 / 36) * pi)
        return direct

    direct = mainDir()


    def Feature(pos, theta):
        # 将之前计算的梯度方向有图像坐标系换算到物体坐标系
        def _theta(x, y):
            if (x < 0 or x >= r) or (y < 0 or y >= c):
                return 0
            dif = angle[x][y] - theta
            return dif if dif > 0 else dif + 2 * pi

        # 双线性插值
        def DB_linear(x, y):
            xx, yy = int(x), int(y)
            dy1, dy2 = y-yy, yy+1-y
            dx1, dx2 = x-xx, xx+1-x
            val = _theta(xx, yy)*dx2*dy2 \
                  + _theta(xx+1, yy)*dx1*dy2 \
                  + _theta(xx, yy+1)*dx2*dy1 \
                  + _theta(xx+1, yy+1)*dx1*dy1
            return val

        y0, x0 = pos
        # 将原图像x轴转到与主方向相同的方向
        H = np.array([cos(theta), sin(theta)])
        V = np.array([-sin(theta), cos(theta)])

        val = []
        # 计算种子点的梯度强度信息
        def cnt(x1, x2, y1, y2, xsign, ysign):
            voting = [0 for i in range(9)]
            for x in range(x1, x2):
                for y in range(y1, y2):
                    dp = [x * xsign, y * ysign]
                    p = H * dp[0] + V * dp[1]
                    bin = int((DB_linear(p[0]+x0, p[1]+y0))//(pi/4) + 1)
                    # 将360°分成8个bin来进行投票
                    if bin > 8:
                        bin = 8
                    voting[bin] += 1
            return voting[1:]

        bins = (r + c) // 150
        for xsign in [-1, 1]:
            for ysign in [-1, 1]:
                val += cnt(0, bins, 0, bins, xsign, ysign)
                val += cnt(bins, bins*2, 0, bins, xsign, ysign)
                val += cnt(bins, bins*2, bins, bins*2, xsign, ysign)
                val += cnt(0, bins, bins, bins*2, xsign, ysign)
        return val

    # 归一化处理
    feature = []
    for i in range(length):
        val = Feature(corners[i], direct[i])
        m = sum(k * k for k in val) ** 0.5
        l = [k / m for k in val]
        feature.append(l)
    # 返回sift描述子，特征点和特征点的个数
    return feature, corners, length

# 合并图片
def Merge(img1, img2):
    h1, w1, a = np.shape(img1)
    h2, w2, a = np.shape(img2)
    if h1 < h2:
        extra = np.array([[[0, 0, 0] for i in range(w1)] for ii in range(h2-h1)])
        img1 = np.vstack([img1, extra])
    elif h1 > h2:
        extra = np.array([[[0, 0, 0] for i in range(w2)] for ii in range(h1-h2)])
        img2 = np.vstack([img2, extra])
    return np.hstack([img1, img2])

# 定义匹配函数
def _Match(threshold):
    matched = []
    for id in range(len(imgset)):
        x = []
        cnt = 0
        for i in range(lt):
            tmp = []
            for j in range(ll[id]):
                sc = np.inner(np.array(ft[i]), np.array(ff[id][j]))
                tmp.append(sc)
            x.append([tmp.index(max(tmp)), max(tmp)])
        for a in range(len(x)):
            b, s = x[a]
            if s < threshold:
                continue
            cnt += 1

        if cnt > 10:
            print("匹配成功 %d !!!" % id)
            matched.append([id, cnt])

        else:
            print("%d 不匹配" % id)
    matched.sort(key=lambda x: x[1], reverse=True)
    for i in matched:
        # cv2.imwrite("match%d.jpg" % i[0], imgset0[i[0]])
        print("统计的描述符为 %d " % i[1])
        cv2.namedWindow("MATCH_RESULT")
        img = Merge(test_image, imgset0[i[0]])
        img = np.array(img, dtype="uint8")
        cv2.imshow("MATCH_RESULT", img)
        cv2.waitKey(0)
        cv2.destroyWindow("MATCH_RESULT")




if __name__ == "__main__":

    ### SIFT ###
    test_image = cv2.imread(r".\corel\images\4120.jpg", 1)
    imgset0 = [cv2.imread(r".\corel\images\%d.jpg" % i, 1) for i in range(350, 450)]
    print(np.shape(test_image))
    # 灰度化
    grey_test_img = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    imgset = [cv2.cvtColor(imgset0[i], cv2.COLOR_BGR2GRAY) for i in range(len(imgset0))]

    ff = []
    cc = []
    ll = []
    ft, ct, lt = Sift(grey_test_img)
    for i in range(len(imgset)):
        f, c, l = Sift(imgset[i])
        ff.append(f)
        cc.append(c)
        ll.append(l)

    w = np.shape(grey_test_img)[1]

    _Match(0.8)




