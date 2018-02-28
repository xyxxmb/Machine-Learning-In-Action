'''
Created on Nov 17, 2017

Regression

@author: mabing
'''
from numpy import *

# 读取文件，文件的第一列值为1.0，代表x0（偏移量），第二列为横坐标x，第三列为纵坐标y
def loadDataSet(fileName):      
    numFeat = len(open(fileName).readline().split('\t')) - 1 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i])) 
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

'''
求解回归系数 w = ((X^T)X)^(-1)(X^T)y
@param xArr：数据集（包括x0=1）
@param yArr：目标值
'''
def standRegres(xArr,yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:  # NumPy线性代数库linalg中的函数det()计算行列式，如果Mat.T*xMat的行列式为0，则不存在X的逆，将会出现错误
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws  # 回归系数w，2行1列(w[0]、w[1])

'''
# 绘制拟合直线
import matplotlib.pyplot as plt
xArr,yArr = loadDataSet('ex0.txt')
ws = standRegres(xArr,yArr)
print(ws)
xMat = mat(xArr)
yMat = mat(yArr)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])
xCopy = xMat.copy()
xCopy.sort(0)  # 将点按升序排列，防止直线上的数据点次序混乱，绘图出现问题
yHat = xCopy*ws  # 做预测，得到 y(^)
ax.plot(xCopy[:,1], yHat) # 绘制拟合曲线
plt.show()
print(corrcoef(yHat.T,yMat)) # 计算相关系数，其中转置保证两个向量都是行向量
'''
'''
结果：
[[ 1.          0.13653777]
 [ 0.13653777  1.        ]]
非对角线上的系数为yHat.T和yMat的相关系数
'''

'''
局部加权线性回归函数
@param testPoint：待预测点
@param xArr：数据集
@param yArr：目标值
@param k：高斯核中参数k，k越大，用于训练回归模型的数据越多
注意：局部加权线性回归函数的缺点在于，对于每一个数据点，都要运行整个数据集
'''
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))  # 创建对角阵，主对角线均为1
    for j in range(m):       # 权重值大小以指数级衰减
        diffMat = testPoint - xMat[j,:]     
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat)) # 回归系数w
    return testPoint * ws  # 返回预测值

# 循环计算每一个数据点的预测值
def lwlrTest(testArr,xArr,yArr,k=1.0):  
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k) # 缺点：对于每一个数据点，都要运行整个数据集
    return yHat

'''
# 测试局部加权线性回归函数
import matplotlib.pyplot as plt
xArr,yArr = loadDataSet('ex0.txt')
yHat = lwlrTest(xArr, xArr, yArr, 0.003) # k=0.003，会出现过拟合
xMat = mat(xArr)
srtInd = xMat[:,1].argsort(0)  # 索引按照升序排列
xSort = xMat[srtInd][:,0,:]  # 将xMat按照升序排列
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xSort[:,1], yHat[srtInd]) # 绘制回归直线
ax.scatter(xMat[:,1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')
plt.show()
'''

'''
残差平方和rss
@param yArr：目标值
@param yHatArr：预测值
'''
def rssError(yArr,yHatArr): # yArr和yHatArr都需要是数组
    return ((yArr-yHatArr)**2).sum()

'''
# 预测鲍鱼的年龄
abX,abY = loadDataSet('abalone.txt')
yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
# 计算3种k值下的rss
print(rssError(abY[0:99], yHat01.T))   # 56.7842091184，误差最小，但容易过拟合
print(rssError(abY[0:99], yHat1.T))    # 429.89056187
print(rssError(abY[0:99], yHat10.T))   # 549.118170883
print()
# 在新数据下的表现
yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)  # 第一个参数为测试点
yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
# 计算3种k值下的新数据点的rss
print(rssError(abY[100:199], yHat01.T))   # 25119.4591112，因为过拟合，所以在新的测试点上误差反而大
print(rssError(abY[100:199], yHat1.T))    # 573.52614419
print(rssError(abY[100:199], yHat10.T))   # 517.571190538
print()
# 和简单的线性回归作比较
ws = standRegres(abX[0:99], abY[0:99])
yHat = mat(abX[100:199]) * ws
print(rssError(abY[100:199], yHat.T.A))  # 518.636315325，其中yHat.T.A将矩阵转化为数组
'''

'''
岭回归（解决特征比样本点还多的问题（n>m），防止不存在矩阵X的逆。现在也用于在估计中加入偏差，从而得到更好的估计）
@param xMat；数据集
@param yMat：目标值
@param lam：λ的值，用于限制所有w之和，通过引入该惩罚项，减少不重要的参数，在统计学中叫做缩减
'''
def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:  # 还需要检查行列式，防止lam设置为0
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T*yMat)
    return ws
    
def ridgeTest(xArr,yArr):
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)    # 计算每一列的均值
    yMat = yMat - yMean     
    # 对特征矩阵X归一化
    xMeans = mean(xMat,0)   # 计算每一列的均值
    xVar = var(xMat,0)      # 按照列计算方差
    xMat = (xMat - xMeans)/xVar  # 归一化，广播
    numTestPts = 30 # 计算30个不同的λ下的权值
    wMat = zeros((numTestPts,shape(xMat)[1]))  # 30个λ下的权值组成的权值矩阵
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat  # 返回权值矩阵

'''
# 测试岭回归函数
abX,abY = loadDataSet('abalone.txt')
ridgeWeights = ridgeTest(abX,abY)
print(ridgeWeights)
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ridgeWeights) # 横坐标为λ，纵坐标(一列8个点)为权值w[0]~w[7]
plt.show()
'''

# 按照列进行归一化（均值为0，方差为1）
def regularize(xMat):
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   # 按列求均值
    inVar = var(inMat,0)  # 按列求方差
    inMat = (inMat - inMeans)/inVar
    return inMat

'''
前向逐步回归：容易找出重要特征，即使停止对那些不重要特征的收集。如果用于测试，可以使用类似于10折交叉验证的方法比较这些模型，选择rss最小的模型
@param xArr：数据集
@param yArr：目标值
@param eps：步长
@param numIt：迭代次数
'''
def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat = mat(xArr); yMat = mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean     # 也可以归一化y，但会得到更小的相关系数
    xMat = regularize(xMat) 
    m,n = shape(xMat)
    returnMat = zeros((numIt,n)) # 记录每次迭代的权值向量，构成一个矩阵 (迭代次数，特征数)
    ws = zeros((n,1)); wsTest = ws.copy(); wsMax = ws.copy()  # 权重初始化
    for i in range(numIt):
        print(ws.T)  
        lowestError = inf;  # 设置当前最小误差为+∞
        for j in range(n):  # 对每个特征 
            for sign in [-1,1]:  # 增大或减小
                wsTest = ws.copy()
                wsTest[j] += eps*sign # 按照一定步长，改变系数，得到一个新的W
                yTest = xMat*wsTest  
                rssE = rssError(yMat.A,yTest.A) # 新W下的误差
                if rssE < lowestError:  # 如果误差小于当前误差，更新误差值
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T
    return returnMat  # 返回迭代矩阵

# 测试前向逐步回归方法
xArr,yArr = loadDataSet('abalone.txt')
returnMat200 = stageWise(xArr,yArr,0.01,200)   # (200,8)，最后一次迭代效果 [[ 0.04  0.    0.09  0.03  0.31 -0.64  0.    0.36]]
returnMat5000 = stageWise(xArr,yArr,0.001,5000) # (5000,8)，最后一次迭代效果 [[ 0.044 -0.011  0.12   0.022  2.023 -0.963 -0.105  0.187]]
# 绘图显示
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(returnMat5000) # 横坐标为迭代次数，纵坐标(一列8个点)为权值w[0]~w[7]
plt.show()
# 与最小二乘法比较，发现和5000次迭代，步长0.001的结果类似
xMat = mat(xArr)
yMat = mat(yArr).T
xMat = regularize(xMat)
yMat = yMat - mean(yMat,0)
weights = standRegres(xMat,yMat.T)
print(weights.T) # [[ 0.0430442  -0.02274163  0.13214087  0.02075182  2.22403814 -0.99895312 -0.11725427  0.16622915]]

from time import sleep
# import socket
import json
import urllib.request as req  # python3应该导入urllib.request模块，而不是urllib/urllib2模块
def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10) # 休眠10秒钟，防止短时间内过多的API调用
    # socket.setdefaulttimeout(10)
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
    pg = req.urlopen(searchURL)
    retDict = json.loads(pg.read()) # 用json.load()方法打开和解析url页面内容，得到一部字典，找出价格和其他信息
    pg.close()
    for i in range(len(retDict['items'])):  # 遍历所有条目
        try:
            currItem = retDict['items'][i]  # 获得当前条目
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else: newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if  sellingPrice > origPrc * 0.5:
                    print("%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except: print('problem with item %d' % i)
    
def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)

'''
交叉验证测试岭回归（用缩减法确定最佳回归系数）：可以观察到缩减程度，同时可以帮助选取主要的特征
@param xArr：数据集，list对象
@param yArr：目标值，list对象
@param numVal：交叉验证的次数
'''
def crossValidation(xArr,yArr,numVal=10):
    m = len(yArr)                           
    indexList = list(range(m))
    errorMat = zeros((numVal,30)) #create error mat 30columns numVal rows
    for i in range(numVal):
        trainX=[]; trainY=[]
        testX = []; testY = []
        random.shuffle(indexList)  # 随机打乱一个索引list
        for j in range(m):
            if j < m*0.9:   # 90%的数据作为训练集
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:  # 10%的数据作为测试集
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX,trainY)    # 得到30个不同λ下的权值向量组成的矩阵
        for k in range(30): #loop over all of the ridge estimates
            matTestX = mat(testX); matTrainX=mat(trainX)
            meanTrain = mean(matTrainX,0)
            varTrain = var(matTrainX,0)
            matTestX = (matTestX-meanTrain)/varTrain # 归一化测试集矩阵
            yEst = matTestX * mat(wMat[k,:]).T + mean(trainY) # 测试岭回归的结果
            errorMat[i,k] = rssError(yEst.T.A,array(testY)) # 存储第i次交叉验证第k个λ下的rss值
            #print(errorMat[i,k])
    meanErrors = mean(errorMat,0) # 计算每一个λ下的平均错误率
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors==minMean)] # nonzero(condition)返回满足条件不为0的下标，找到最小权值向量
    # can unregularize to get model
    # when we regularized we wrote Xreg = (x-meanX)/var(x)
    # we can now write in terms of x not Xreg:  x*w/var(x) - meanX/var(x) + meanY
    xMat = mat(xArr); yMat=mat(yArr).T
    meanX = mean(xMat,0); varX = var(xMat,0)
    unReg = bestWeights/varX  # 因为岭回归使用了归一化，使用要数据还原，即非归一化下的权值
    print("the best model from Ridge Regression is:\n",unReg)  # 非归一化的权值
    print("with constant term: ",-1*sum(multiply(meanX,unReg)) + mean(yMat)) # 求解常数项 x0 = -wx + y

'''
# 由于乐高玩具的API已经关闭，故setDataCollect()无法爬取数据，故无法测试这个例子
lgX = []
lgY = []
setDataCollect(lgX, lgY) # python中参数是传引用调用，故lgX, lgY会修改
print(shape(lgX)) # (58,4)
# 将特征 X0=1 加入第1列
lgX1 = mat(ones(58, 5))
lgX1[:,1:5] = mat(lgX)
ws = standRegres(lgX1,lgY) # 最小二乘法回归处理得到回归系数
print(lgX1[0] * ws) # 检查预测结果
print(lgX1[-1] * ws) # 检查预测结果
crossValidation(lgX,lgY,10) # 交叉验证岭回归得到最好的回归系数
'''

