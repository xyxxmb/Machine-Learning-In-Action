'''
Created on Nov 19, 2017

k Means Clustering for Ch10 of Machine Learning in Action

@author: mabing
'''
from numpy import *

def loadDataSet(fileName):    # 注意，这里读取的最后一行不是目标值，因为是无监督学习
    dataMat = []                
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine)) # 映射一行元素为float
        dataMat.append(fltLine)
    return dataMat

# 计算欧氏距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) 

# 为数据集中每个特征随机创建k个聚簇中心
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))  # k个聚簇中心，每个簇中心特征数为n
    for j in range(n):
        minJ = min(dataSet[:,j]) 
        rangeJ = float(max(dataSet[:,j]) - minJ)
        # 为第j个特征创建k个聚簇中心
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1)) # random.rand(k,1) 是numpy中的函数，随机生成k行1列的(0,1)范围的高斯随机数
    return centroids

'''
# 测试函数功能
datMat = mat(loadDataSet('testSet.txt'))
print(distEclud(datMat[0],datMat[1]))  # 欧氏距离 5.18463281668
print(randCent(datMat, 5)) # (5,2)，5个聚簇中心，2个特征
'''

'''
kMeans算法
@param dataSet：数据集
@param k：k个聚类中心
@param distMeas：距离计算公式，默认欧氏距离计算法
@param createCent：选择k个聚类中心方法，默认为随机选取
'''
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))  # 存储每个点的簇分配结果，第一列记录簇索引值，第二列存储误差（距离的平方）
    centroids = createCent(dataSet, k)  # k个聚簇中心矩阵(k,n)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m): # 对于每个数据点
            minDist = inf; minIndex = -1
            for j in range(k):  # 寻找最近的质心
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True # 如果任何一个簇中心位置发生了改变，那么就更改标志clusterChanged为true
            clusterAssment[i,:] = minIndex,minDist**2
        print(centroids)
        for cent in range(k): # 更新质心的位置
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]] # 得到在这个簇中心的所有数据点
            centroids[cent,:] = mean(ptsInClust, axis=0) # 按列求均值（注意：簇中心不包括随机选择的那个点）
    return centroids, clusterAssment # 返回簇中心、簇分配结果矩阵

'''
# 测试kMeans算法
datMat = mat(loadDataSet('testSet.txt'))
myCentroids, clustAssing = kMeans(datMat,4)
print(myCentroids) # 经过数次迭代后会收敛
print(clustAssing) # 第1列为簇索引，第2列为每个点到簇中心的误差
'''

'''
二分K均值算法：克服K-均值算法收敛于局部最小值的问题
@param dataSet：数据集
@param k：指定的k个聚类中心
@param distMeas：距离计算公式，默认欧氏距离计算法
'''
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]  # 创建一个初始簇，只有一个簇中心，并转化为列表
    centList =[centroid0] # 加一层列表包含初始簇列表，以后会越来越多，直到等于k个
    for j in range(m): # 计算初始误差
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k): # 簇中心的数目小于k个
        lowestSSE = inf
        for i in range(len(centList)):  # 对每一个簇划分
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:] # 得到在簇i中的数据点
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas) # 使用kMeans算法将簇一分为二
            sseSplit = sum(splitClustAss[:,1]) # 计算新划分的簇中数据的SSE（误差平方和）
            # 计算剩余簇中数据的SSE值，如果是第一次，该值为0，因为所有数据都用来划分新簇，故没有剩余簇的数据
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])  
            print("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:  # 两部分SSE作为总误差
                bestCentToSplit = i   # 最好的簇划分，即按照第i个簇划分
                bestNewCents = centroidMat.copy()  # 新簇，在原来基础上+1个簇中心，要用copy方法，防止矩阵传引用同时更改bestNewCents的值
                bestClustAss = splitClustAss.copy() # 新的划分结果，要用copy方法，防止矩阵传引用同时更改bestClustAss的值
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) # 为新划分出的簇赋一个新的编号，如1,2,3,...
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print('the bestCentToSplit is: ',bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0] # 更新原来的旧簇的值
        centList.append(bestNewCents[1,:].tolist()[0]) # 将新划分出的簇加入centList中
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClustAss # 更新clusterAssment
    return mat(centList), clusterAssment  # 返回簇中心、簇分配结果矩阵

'''
# 测试二分K均值算法
datMat3 = mat(loadDataSet('testSet2.txt'))
centList, myNewAssment = biKmeans(datMat3,3)
print(centList)
print(myNewAssment)
'''

# 实例：对于地理数据应用二分K-均值算法并绘图

# 抓取获取地理位置的API（访问不了，会报502错误）
import urllib.request as req
import urllib.parse as par
import json
def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'  #create a dict and constants for the goecoder
    params = {}
    params['flags'] = 'J' # 返回类型为JSON
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = par.urlencode(params)
    yahooApi = apiStem + url_params    # 打印url参数
    print(yahooApi)
    c = req.urlopen(yahooApi)  # 现在打不开了，报502网关错误，不过好在所要的数据已经保存到 place.txt 文件中
    return json.loads(c.read())

# 解析JSon数据，得到位置的经纬度，将地址和位置信息写入到 place.txt 文件
from time import sleep
def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print("%s\t%f\t%f" % (lineArr[0], lat, lng))
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else: print("error fetching")
        sleep(1)
    fw.close()
 
# 球面距离计算   
def distSLC(vecA, vecB): # vecA，vecB 是两个经纬度向量
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)   # pi在numpy中被导入
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * \
                      cos(pi * (vecB[0,0]-vecA[0,0]) /180)
    return arccos(a + b)*6371.0 # 返回地球表面两点间的距离

# 簇绘图函数，默认为5个聚类中心，可以按情况修改
import matplotlib.pyplot as plt
def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])]) # 每一个位置的经纬度组成一个列表
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)  # 使用二分均值算法获得聚簇中心和簇分配结果
    fig = plt.figure()
    rect = [0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']   # 定义很多个不同的标记形状
    axprops = dict(xticks=[], yticks=[]) 
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')  # 基于一副图像创建矩阵
    ax0.imshow(imgP)  # 绘制图形矩阵
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)  # 另一套坐标系，绘制数据点和聚类中心
    for i in range(numClust):  # 对于每一个簇
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]  # 选择满足第i个簇的所有数据点
        markerStyle = scatterMarkers[i % len(scatterMarkers)]  # 选择标记形状
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90) # 绘制数据点
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)  # 绘制簇中心
    plt.show() # 显示，最后将绘制结果保存为 result.png

# 对于地理数据应用二分K-均值算法并绘图
clusterClubs()
