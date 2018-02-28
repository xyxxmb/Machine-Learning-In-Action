'''
Created on Oct 30, 2017

Logistic Regression Working Module

@author: mabing
'''
from numpy import *

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])  # X0的值初始化设置为1.0
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

'''
Logistic 回归梯度上升优化算法
@param dataMatIn：特征矩阵，行代表样本数，列代表特征数
@param classLabels：每个样本对应的类标签组成的向量
'''
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)             # 转化为 NumPy 矩阵
    labelMat = mat(classLabels).transpose() # 转化为 NumPy 矩阵
    m,n = shape(dataMatrix)  # m=100，n=3，代表100个样本，3个特征（包括X0）
    alpha = 0.001  # 步长，学习速率
    maxCycles = 500 # 最大迭代次数
    weights = ones((n,1))
    for k in range(maxCycles):          # 最大迭代次数
        h = sigmoid(dataMatrix*weights) # 如果两参数都是矩阵，那么*和dot()都为矩阵相乘，而如果两个都是数组，则*为对应位置相乘，dot()为矩阵相乘
        error = (labelMat - h)          # 误差
        weights = weights + alpha * dataMatrix.transpose() * error # 更新权值
    return weights

'''
# 测试求得的最优回归系数
dataArr, labelMat = loadDataSet()
weights = gradAscent(dataArr, labelMat)
print(weights)
'''

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    print(weights)
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s') # 画散点图
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1) # 步长0.1
    y = (-weights[0]-weights[1]*x)/weights[2]  # 0=w0x0+w1x1+w2x2，其中x0=1，y即x2
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

'''
# 测试画图
dataArr, labelMat = loadDataSet()
weights = gradAscent(dataArr, labelMat)
plotBestFit(weights.getA()) # 注意：getA() 方法将自身返回成一个n维数组对象，否则 weights[1] 将为[[0.48007329]]，而不是0.48007329，故会报错
'''

'''
随机梯度上升算法，一次仅用一个样本点来更新回归系数，是一种"在线学习"算法，即在样本到来时实现分类器的增量式更新
@param dataMatIn：特征矩阵，行代表样本数，列代表特征数
@param classLabels：每个样本对应的类标签组成的向量
注意，随机梯度上升算法的好处是可以避免上述方法每次迭代都要对整个数据集进行操作，而且不涉及矩阵操作
'''
def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   # 初始化权值为1，这里是列表，不是矩阵
    for i in range(m):  # 对于每个样本
        h = sigmoid(sum(dataMatrix[i]*weights))  # 数值，不是向量
        error = classLabels[i] - h   # 数值，不是向量
        weights = weights + alpha * error * dataMatrix[i]  # 对于每个样本进行权值的更新
    return weights

'''
# 随机梯度算法测试画图
dataArr, labelMat = loadDataSet()
weights = stocGradAscent0(array(dataArr), labelMat)  # 必须将dataArr转化为数组，不然alpha * error * dataMatrix[i]作用到矩阵上回报错
plotBestFit(weights) # 注意：这里weight是一个列表，不是一个矩阵，故不用getA()方法
'''

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   # 数组，[1, n]
    for j in range(numIter):  # 改进1：增加迭代次数，j为迭代次数
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    # 改进2：alpha随迭代次数每次减少 1/(j+i)
            randIndex = int(random.uniform(0,len(dataIndex))) # 改进3：随机选取样本来更新权值系数，较少周期性波动
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])  # 要删除选择的该样本
    return weights # [1, n]

'''
# 改进后的随机梯度算法测试画图
dataArr, labelMat = loadDataSet()
weights = stocGradAscent1(array(dataArr), labelMat)  # 必须将dataArr转化为数组，不然alpha * error * dataMatrix[i]作用到矩阵上回报错
plotBestFit(weights) # 注意：这里weight是一个列表，不是一个矩阵，故不用getA()方法
'''

'''
使用Logistic回归估计马疝气病的死亡率
@param inX：测试向量，包含所有特征，如 [1, 21]
@param weights：训练的回归系数，即最优权值，如 [1, 21]
'''
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights)) # 对应位置相乘，即 σ(z) = σ(x0w0 + x1w1 + x2w2 + ...)
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):  # 21 个特征，最后一个是分类标签，0 和 1 两类
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))

multiTest() # 大概35%错误率，但是结果并不差，因为原数据集有30%数据缺失，缺失部分用0代替了
# 可以调整colicTest()中的迭代次数（如500->1000）和stocGradAscent1()中的步长（alpha的值），可以降低至20%左右