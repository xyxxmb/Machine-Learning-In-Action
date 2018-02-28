'''
Created on Nov 16, 2017

Adaboost is short for Adaptive Boosting

@author: mabing
'''
from numpy import *

def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

'''
通过阈值比较对数据进行分类
@param dataMatrix：数据集
@param dimen：第i个特征
@param threshVal：阈值
@param threshIneq：不等号(＜、＞)
'''
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq): 
    retArray = ones((shape(dataMatrix)[0],1))  # 初始化为1
    if threshIneq == 'lt':  
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0  # 数组过滤
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray  # 返回第i个特征的的分类结果
    
'''
遍历stumpClassify()函数所有可能的输入值，并找到数据集上最佳的单层决策树
@param dataArr：数据集
@param classLabels：标签
@param D：权值向量
'''
def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {} # 存储给定向量D时所得到的的最佳单层决策树的相关信息
    bestClasEst = mat(zeros((m,1)))
    minError = inf  # 初始的最小错误率为 +∞
    for i in range(n):  # 对数据集的每一个特征
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1): # 从最小值到最大值滑动，步长为0.1
            for inequal in ['lt', 'gt']: # 对于＜和＞两种情况
                threshVal = (rangeMin + float(j) * stepSize) 
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal) # 预测的分类标签
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr  # 计算错误率
                print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy() 
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
                    # print(bestStump, minError, bestClasEst)
    return bestStump,minError,bestClasEst  # 返回所得到的的最佳单层决策树的相关信息、最小错误率、最好的分类标签

'''
# 测试一个弱分类器（单层决策树）的产生过程
dataMat, classLabels = loadSimpData()
D = mat(ones((5,1))/5)  # 权值向量，归一化
bestStump, minError, bestClasEst = buildStump(dataMat, classLabels, D)
print(bestStump, minError, bestClasEst)

'''
'''
输出结果：
{'dim': 0, 'thresh': 1.3, 'ineq': 'lt'} [[ 0.2]] [[-1.]
 [ 1.]
 [-1.]
 [-1.]
 [ 1.]]
分析：根据的是第1个特征进行分类，阈值为1.3时，对于＜这种情况，分类的错误率为0.2，对应的最好的分类标签为 [-1,1,-1,-1,1]
'''

'''
基于单层决策树的AdaBoost训练过程
@param dataArr：数据集
@param classLabels：标签
@param numIt：迭代次数
'''
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr = []  # 存储每个弱分类器（单层决策树）的参数
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)   # 初始化权值向量，刚开始都相同
    aggClassEst = mat(zeros((m,1))) # 记录每个样本数据点的预测类别的累加值，非常重要！！！
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D) # 构建单层决策树
        print("D:",D.T) # 输出每个样本权值
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))  # 计算每个弱分类器的α值, max(error,eps)防止除0溢出，如果该分类器效果越好，α越大
        bestStump['alpha'] = alpha  
        weakClassArr.append(bestStump)      # 存储每个弱分类器（单层决策树）的参数
        print("classEst: ",classEst.T)  # 每个样本的预测类别
        expon = multiply(-1*alpha*mat(classLabels).T,classEst) # 计算指数，对应元素相乘
        D = multiply(D,exp(expon))    # 计算新的权值向量用于下一次迭代
        D = D/D.sum() 
        # aggClassEst 表示当前迭代次数下的每个样本数据点预测类别的累加值，要与真实类别比较，故非常重要！！！
        aggClassEst += alpha*classEst  
        print("aggClassEst: ",aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1))) 
        errorRate = aggErrors.sum()/m
        print("total error: ",errorRate)
        if errorRate == 0.0: break  # 如果错误率为0，提前退出迭代过程
    return weakClassArr,aggClassEst  # 返回存储每个弱分类器（单层决策树）参数的数组和每个样本数据点预测类别的累加值

'''
# 测试AdaBoost
dataMat, classLabels = loadSimpData()
weakClassArr,aggClassEst = adaBoostTrainDS(dataMat, classLabels,9) # 实际上在达到第3次迭代时分类的error=0，就会自动退出迭代过程
print(weakClassArr)
print(aggClassEst)
'''

'''
@param datToClass：测试的数据点
@param classifierArr：存储每个弱分类器（单层决策树）参数的数组
'''
def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass) # 转化为NumPy矩阵
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])  # 第i个弱分类器的预测类别
        aggClassEst += classifierArr[i]['alpha']*classEst
        print(aggClassEst)  # 输出每次弱分类器迭代的累加值结果
    return sign(aggClassEst) # 返回测试数据点类别

'''
# 测试分类结果
print('\n')
adaClassify([[5,5],[0,0]],weakClassArr)
'''
'''
结果
[[ 0.69314718]
 [-0.69314718]]
[[ 1.66610226]
 [-1.66610226]]
[[ 2.56198199]
 [-2.56198199]]
发现数据点[5,5]分为类别1越来越强，数据点[0,0]分为类别-1越来越强
'''

def loadDataSet(fileName):   # 读取文件
    numFeat = len(open(fileName).readline().split('\t')) 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))  # 最后一个是类别
    return dataMat,labelMat

'''
ROC曲线（接收者操作特征）的绘制及AUC（曲线下的面积）计算函数
@param predStrengths：Numpy数组或行向量组成的矩阵，代表每个分类器的预测强度
@param classLabels：真实的类标签
'''
def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0) #cursor
    ySum = 0.0 #variable to calculate AUC
    numPosClas = sum(array(classLabels)==1.0)
    yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndicies = predStrengths.argsort() # 得到矩阵中每个元素的排序索引（从小大大）
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    #loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:  # 将矩阵转化为列表
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep;
        else:
            delX = xStep; delY = 0;
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print("the Area Under the Curve is: ",ySum*xStep)  # 计算AUC，完美分类器的AUC（实线）为1.0，预测AUC（虚线）为0.5

# 在一个难数据集上应用AdaBoost
# 训练阶段
dataMat,labelMat = loadDataSet('horseColicTraining2.txt')
classifierArr, aggClassEst = adaBoostTrainDS(dataMat,labelMat,10)  # total error：≈23%
# 绘制ROC曲线
plotROC(aggClassEst.T, labelMat)  # AUC = 0.8582969635063604
# 测试阶段
testDataMat,testLabelMat = loadDataSet('horseColicTest2.txt')
prediction10 = adaClassify(testDataMat,classifierArr)
errArr = mat(ones((67,1)))  # 测试集中有67组数据
errNum = errArr[prediction10 != mat(testLabelMat).T].sum() # 统计错误分类数量
print(errNum) # 16个错误，错误率除以67即可
