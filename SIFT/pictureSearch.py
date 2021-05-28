# coding:utf-8
import cv2
import numpy as np


# 寻找相似图片
def search_img(test_image, imgset, method):
    kp1, des1 = method.detectAndCompute(test_image, None)  # kp1 关键点列表 des1 关键点描述符
    close = float("inf")    # 最近距离初始化为无穷大
    similar = None     # 记录最相似的图片
    print("开始匹配..................")
    for image in imgset:
        kp2, des2 = method.detectAndCompute(image, None)
        if des2 is not None:
            bf = cv2.BFMatcher()
            dis = 0  # 储存两图间所有匹配关键点的总距离
            matches = bf.match(des1, des2)  # 寻找匹配对
            for match in matches:   # 将匹配对欧氏距离总和作为衡量相似度的标准
                dis += match.distance
            if dis < close: # 更新最相似图片
                close = dis
                similar = image
    print("匹配结束")

    return similar

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


if __name__ == "__main__":
    # 加载数据
    imgset = [cv2.imread(r'.\corel\images\%d.jpg' % i, 1) for i in range(1000)]
    test_image = cv2.imread(r".\corel\images\100.jpg", 1)

    #sift = cv2.xfeatures2d.SIFT_create()
    orb = cv2.ORB_create()
    similar_image_ORB = search_img(test_image, imgset, orb)
    imgs = np.array(Merge(test_image, similar_image_ORB), dtype="uint8")
    cv2.imshow("compare", imgs)     # 同窗口测试图片以及在训练集找到的图片
    cv2.waitKey(0)
