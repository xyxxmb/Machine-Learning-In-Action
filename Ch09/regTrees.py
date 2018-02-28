'''
Created on Nov 18, 2017

Tree-Based Regression Methods

@author: mabing
'''
from numpy import *

def loadDataSet(fileName):     
    dataMat = []                # 最后一列是目标值
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine)) # 将每一行映射为浮点数
        dataMat.append(fltLine)
    return dataMat

'''
按照特征和特征值将数据集划分为两个子集
@param dataSet：数据集合
@param feature：特征
@param value：特征值
'''
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0,mat1

'''
# 测试划分数据集函数 binSplitDataSet()
testMat = mat(eye(4))  # 单位矩阵
mat0,mat1 = binSplitDataSet(testMat,1,0.5)
print(mat0)
print(mat1)
'''

# 返回叶节点，叶节点为目标值的均值
def regLeaf(dataSet):
    return mean(dataSet[:,-1])

# 计算目标值的总方差 = (均)方差 * 样本数目
def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]

'''
用最佳方式切分数据集并生成相应的叶节点：中间包括3个预剪枝操作（①②③）
@param dataSet：数据集，递归调用后变为子数据集
@param leafType：叶节点，目标值的均值
@param errType：目标值的总方差
@param ops：可选参数元组，包括容许的误差下降值tolS和切分的最少样本数tolN，其中tolS对误差的数量级很敏感，不太好控制
'''
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0]  # 容许的误差下降值为1
    tolN = ops[1]  # 切分的最少样本数为4
    # 统计目标值中不相等的数目，如果该数目为1，则不需要切分而直接返回
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: 
        return None, leafType(dataSet) # ① 特征返回None，特征值返回目标值的均值
    m,n = shape(dataSet)
    S = errType(dataSet)  # 目标值总方差
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1): # 对每个特征（不包括最后一列的目标值）
        for splitVal in set(dataSet[:,featIndex].T.tolist()[0]): # 对每个不重复的特征值
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)  # 按照特征和特征值划分成两个数据子集
            # 如果切分的数据集很小，则直接返回，即放弃这种切分方法
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS: 
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # ② 如果误差减小的不太大，则直接返回
    if (S - bestS) < tolS: 
        return None, leafType(dataSet) # 特征返回None，特征值返回目标值的均值
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # ③ 如果切分的数据集很小，则直接返回，即不应切分
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  
        return None, leafType(dataSet)  # 特征返回None，特征值返回目标值的均值
    return bestIndex, bestValue  # 返回最好的切分特征和特征值

# 创建CART回归树，树结构用一个字典存储
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops) # 选择最好的分割方式，最重要的函数
    if feat == None: return val # 如果返回的特征为None，则直接返回目标值的均值，作为叶子
    retTree = {}
    retTree['spInd'] = feat  # 特征
    retTree['spVal'] = val   # 特征值
    # 划分为左子树和右子树，递归创建树
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree  

'''
# 测试CART回归树
'''
myDat = loadDataSet('ex00.txt')
myMat = mat(myDat)
retTree = createTree(myMat)
print(retTree)  # {'spInd': 0, 'spVal': 0.48813, 'left': 1.0180967672413792, 'right': -0.044650285714285719}
print()
# 另一个较复杂的数据集上
myDat = loadDataSet('ex0.txt')
myMat = mat(myDat)
retTree = createTree(myMat)
print(retTree)  
'''
结果：包含5个叶节点
{'spInd': 1, 'spVal': 0.39435, 'left': {'spInd': 1, 'spVal': 0.582002, 'left': {'spInd': 1, 'spVal': 0.797583
, 'left': 3.9871631999999999, 'right': 2.9836209534883724}, 'right': 1.980035071428571}, 'right': {'spInd': 1
, 'spVal': 0.197834, 'left': 1.0289583666666666, 'right': -0.023838155555555553}}
'''

# 判断一个结点是否是一棵子树
def isTree(obj):
    return (type(obj).__name__=='dict')

# 递归函数，从上往下遍历树直到叶节点为止。如果找到两个叶节点，则计算它们的平均值
def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0

'''
# 回归树后剪枝函数  
@param tree：待剪枝的树
@param testData：剪枝所需的测试数据
''' 
def prune(tree, testData):
    if shape(testData)[0] == 0: return getMean(tree)    # 如果没有测试数据，则对树进行塌陷处理
    if (isTree(tree['right']) or isTree(tree['left'])): # 如果存在子树，要进行裁剪操作
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])  # 将测试数据分为两个子集
    # 递归调用，对测试数据进行切分
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)  
    if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):  # 如果都是叶子
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) +\
            sum(power(rSet[:,-1] - tree['right'],2))  # 不合并的误差
        treeMean = (tree['left']+tree['right'])/2.0   # 左右叶节点按均值合并
        errorMerge = sum(power(testData[:,-1] - treeMean,2))  # 合并的误差
        if errorMerge < errorNoMerge:  # 如果合并的误差比不合并的误差要小，就将左右叶节点合并
            print("merging")
            return treeMean
        else:  # 否则，该两个叶节点不合并，直接返回（待剪枝的树）
            return tree 
    else: return tree 

# 测试后剪枝函数
myDat2 = loadDataSet('ex2.txt')
myMat2 = mat(myDat2)
myTree = createTree(myMat2, ops=(0,1)) # tolS=0，tolN=1，为了创建所有可能中最大的树，便于测试后剪枝
# 导入测试数据
myDatTest = loadDataSet('ex2test.txt')
myMat2Test = mat(myDatTest)
# 执行剪枝过程
newTree = prune(myTree, myMat2Test)
print(newTree)  # 剪枝后的树


# 模型树的叶节点生成函数

# 将数据集格式化为自变量X和目标变量Y，同时计算回归系数ws
def linearSolve(dataSet):  
    m,n = shape(dataSet)
    X = mat(ones((m,n))); Y = mat(ones((m,1)))
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

# 生成叶节点，叶节点是一个回归系数向量
def modelLeaf(dataSet):
    ws,X,Y = linearSolve(dataSet)
    return ws

# 计算误差 = (目标值-预测值)的平方，然后求和
def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))

# 测试模型树
print()
myMat2 = mat(loadDataSet('exp2.txt'))
print(createTree(myMat2, modelLeaf, modelErr, (1,10))) 
'''
结果：以0.285477为界限，分为两段线性模型，第一段 y = 3.468 + 1.1852x，第二段 y = 1.698 + 1.196x
{'spInd': 0, 'spVal': 0.285477, 'left': matrix([[  1.69855694e-03],
        [  1.19647739e+01]]), 'right': matrix([[ 3.46877936],
        [ 1.18521743]])}
'''

# 对回归树叶节点进行预测，model是叶节点值，inDat是待测试数据
def regTreeEval(model, inDat):
    return float(model) #返回叶节点值

# 对模型树叶节点进行预测，model是回归系数ws向量，inDat是待测试数据
def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1,n+1))) # 第一列为偏移量x0=1
    X[:,1:n+1] = inDat
    return float(X*model) # 返回预测值

# 对一个数据点进行预测，返回一个浮点值
def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']: # 如果测试数据点比最优划分的数据点大，则遍历左子树
        if isTree(tree['left']): return treeForeCast(tree['left'], inData, modelEval)
        else: return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']): return treeForeCast(tree['right'], inData, modelEval)
        else: return modelEval(tree['right'], inData)

# 遍历每一个数据点，得到预测值     
def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, mat(testData[i]), modelEval)  # 遍历每一个数据点
    return yHat

# 计算相关系数corrcoef()比较树回归、模型树与标准回归
trainMat = mat(loadDataSet('bikeSpeedVsIq_train.txt'))
testMat = mat(loadDataSet('bikeSpeedVsIq_test.txt'))

myTree = createTree(trainMat, ops=(1,20)) # 创建CART回归树
yHat = createForeCast(myTree, testMat[:,0])  # 得到预测值
print(corrcoef(yHat, testMat[:,1], rowvar=0)[0,1]) # 计算回归树的相关系数，R^2 = 0.964085231822，接近1，很好

myTree = createTree(trainMat, modelLeaf, modelErr, ops=(1,20)) # 创建模型树
yHat = createForeCast(myTree, testMat[:,0], modelTreeEval)  # 得到预测值
print(corrcoef(yHat, testMat[:,1], rowvar=0)[0,1]) # 计算模型树的相关系数，R^2 = 0.976041219138，接近1，比CART回归树还要好

# 标准的线性回归
ws,X,Y = linearSolve(trainMat)
print(ws)  # 标准回归的系数值
'''
ws的结果
[[ 37.58916794]
 [  6.18978355]]
'''
for i in range(shape(testMat)[0]):
    yHat[i] = testMat[i,0] * ws[1,0] + ws[0,0]
print(corrcoef(yHat, testMat[:,1], rowvar=0)[0,1]) # 计算标准回归的相关系数，R^2 = 0.943468423567，接近1，但效果没有两种树回归好