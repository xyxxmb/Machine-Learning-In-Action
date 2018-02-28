'''
Created on Oct 29, 2017

ID3

@author: mabing
'''

from math import log
import operator

'''
计算给定数据集的信息熵
@param dataSet：数据集（包括类别）
'''
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {} # key：类别，value：类别出现的次数
    for featVec in dataSet: # 遍历数据集，统计类别出现的次数
        currentLabel = featVec[-1]  # 得到类别
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries  # 选择该分类的概率 p(xi) = 类别次数/总次数
        shannonEnt -= prob * log(prob,2) # 所有类别的信息熵 H = -p(xi) * log2(p(xi))
    return shannonEnt

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels

'''
# 测试信息熵
dataSet, labels = createDataSet()
# dataSet[0][-1] = 'maybe'   # 把第一行的最后一个分类类别换为'maybe'，则熵会变大。熵越大，混合的数据也越多，数据越无序
print(dataSet)
shannonEnt = calcShannonEnt(dataSet)
print(shannonEnt)
'''

'''
按照给定特征划分数据集
@param dataSet：数据集（包含分类标签）
@param axis：第axis个特征
@param value：希望该特征的返回值
'''
def splitDataSet(dataSet, axis, value):
    retDataSet = []   # 因为python传递的是列表的引用，故需要新建一个数据集，不然在函数内操作会修改dataSet
    for featVec in dataSet:
        if featVec[axis] == value:  # 如果该特征值和期望的值相同
        	# 将符合该特征的数据抽取出来，即从featerVec中去掉该特征axis，即[0,axis) [axis+1:
            reducedFeatVec = featVec[:axis]     
            reducedFeatVec.extend(featVec[axis+1:]) # extend是把列表所有元素与之前合并，而append是把列表当做一个整体与之前合并，故不能用append
            retDataSet.append(reducedFeatVec) # 把抽取掉axis特征的数据添加到结果集中
    return retDataSet

'''
# 测试划分数据集
dataSet, labels = createDataSet()
retDataSet = splitDataSet(dataSet, 0, 1)  # 根据第1个特征的值为1，进行划分
print(retDataSet)   # [[1, 'yes'], [1, 'yes'], [0, 'no']]
retDataSet = splitDataSet(dataSet, 1, 0)  # 根据第2个特征的值为0，进行划分
print(retDataSet)  # [[1, 'no']]，即 [1, 0, 'no'] 抽取掉第2个特征(0)后得到的结果
'''

'''
选择最好的特征作为根(或非叶子结点)进行划分
@param dataSet：数据集（包含分类标签）
'''
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      # 得到特征数（不包括最后一列的分类标签）
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):       
        featList = [example[i] for example in dataSet]  # dataSet中第i个特征所有取值
        # print(featList)   # [1, 1, 1, 0, 0]  [1, 1, 0, 1, 1]
        uniqueVals = set(featList)       # 得到第i个特征里，独一无二的取值集合
        newEntropy = 0.0
        for value in uniqueVals:  # 对于第i个特征的每个唯一取值，进行数据集划分
            subDataSet = splitDataSet(dataSet, i, value)  # 从dataSet中抽取出该特征
            # 求数据集划分后的新信息熵
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)   # 对所有唯一特征值得到的熵求和 
        infoGain = baseEntropy - newEntropy     # 得到信息增益，即熵的减小量
        if (infoGain > bestInfoGain):       # 选择信息增益最大的划分
            bestInfoGain = infoGain         
            bestFeature = i  # 第i个特征就是能使信息增益达到最大的一个划分
    return bestFeature    # 返回最好的特征

'''
# 测试选择最好特征值进行划分
dataSet, labels = createDataSet()
bestFeature = chooseBestFeatureToSplit(dataSet)  # 根据第1个特征的值为1，进行划分
print(bestFeature)  # 0，代表用第1个特征划分，可以使信息增益最大
'''

'''
统计每个类标签出现的频率
@param classList：类别列表，如 ['yes', 'no', 'yes']
'''
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) # 按频率从大到小排序
    return sortedClassCount[0][0]  # 出现频数最多的类的名称，如 'yes'

'''
递归创建树的代码
@param dataSet：数据集，每递归一次，数据集中的特征数少1个（含有分类标签）
@param labels：数据集中特征对应的名称（如 [x1, x2] = ['no surfacing', 'flippers']），每递归一次，特征少1个（和dataSet保持同步）
注意：dataSet删除特征是根据划分splitDataSet()函数减少1个特征个数，labels是直接用del(labels)减少1个特征个数
'''
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    # print(classList[0])
    if classList.count(classList[0]) == len(classList): # 当所有类都相同时，停止分裂，返回该类别标签
        return classList[0]   
    if len(dataSet[0]) == 1:  # 使用完了所有特征，仍然不能将数据集划分为仅包含唯一类别的分组，此时采用投票方式返回频率最高的类别
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]  # 挑选出的最好特征对应的标签，如 0 -> No Surfacing
    # print(bestFeatLabel)
    myTree = {bestFeatLabel:{}}  # 嵌套字典
    # print(myTree)
    del(labels[bestFeat])  # 重要，将lables中该特征删除
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    # print(uniqueVals)
    for value in uniqueVals:
        subLabels = labels[:]   # 拷贝标签列表到新列表，因为python为传引用，不这样做可能会修改原标签列表lables（存在del删除分类标签的操作）
        #print(bestFeatLabel,value)
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels) # 按照最好特征划分数据集，然后递归调用
        #print(myTree)
    return myTree          

'''
# 测试创建树的代码
dataSet, labels = createDataSet()  # labels = ['no surfacing', 'flippers']
myTree = createTree(dataSet, labels)
print(myTree)  # {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
# 导入使用Matplotlib画的这棵决策树
import treePlotter  # 先导入treePlotter.py文件
treePlotter.createPlot(myTree)  # 画图
'''

'''
递归求得分类类别
@param inputTree：创建的一棵训练树，递归一次，是一棵以某一个特征为根结点的子树
@param featLabels：特征标签，如 ['no surfacing', 'flippers']，传递它便于帮我们确定特征在数据集中的位置
@param testVec：测试向量，如 [1, 0]，代表 'no surfacing' = 'yes' 和 'flippers' = 'no'
'''
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]  # inputTree.keys() 返回一个可迭代的对象dict_keys，故要转化为list，然后取出第1个key
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    # print(featIndex)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):  
        classLabel = classify(valueOfFeat, featLabels, testVec)  # 如果还是一个字典类型，则继续递归，即碰到了一个非叶子结点
    else: classLabel = valueOfFeat  # 如果不是字典，则直接返回分类类别，即碰到了叶子结点
    return classLabel

'''
# 测试分类类别
dataSet, labels = createDataSet()  # labels = ['no surfacing', 'flippers']
subLabels = labels[:]  # 注意要用切片形式拷贝
myTree = createTree(dataSet, subLabels)  # 这个函数会改变lables标签，所以先保存一份
print(myTree)  # {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
classLabel = classify(myTree, labels, [1, 1])  # 'yes'
print(classLabel)
classLabel = classify(myTree, labels, [0, 0])  # 'no'
print(classLabel)
'''

'''
使用python的pickle模块存储决策树，写入到指定文件
@param inputTree：决策树
@param filename：写入的文件
注意：pickle存储方式默认是二进制方式
'''
def storeTree(inputTree,filename):
    import pickle   # 要先导入pickle模块
    fw = open(filename,'wb') # 以二进制写的方式创建文件
    pickle.dump(inputTree,fw)
    fw.close()

'''
使用python的pickle模块加载文件，取出决策树
@param filename：读取存储了决策树的文件
''' 
def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)

'''
# 测试pickle读取决策树
dataSet, labels = createDataSet()  # labels = ['no surfacing', 'flippers']
myTree = createTree(dataSet, labels)  # {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
storeTree(myTree, 'myTree.txt')  # 'myTree.txt' 以二进制方式存储，故是乱码
readTree = grabTree('myTree.txt')
print(readTree) # {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
'''

# 使用决策树预测隐形眼镜类型
fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]  # 读一行，去掉首尾空格，再用'\t'分隔各个字段
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesLabelsTem = lensesLabels[:]
lensesTree = createTree(lenses, lensesLabelsTem)  # 这个lensesLabelsTem会在代码中改变，故不能传递lensesLabels，因为之后还要用
print(lensesTree) # 打印这棵树
# 导入使用Matplotlib画的这棵决策树
import treePlotter  # 先导入treePlotter.py文件
treePlotter.createPlot(lensesTree)  # 画图
# 做预测
classLabel = classify(lensesTree, lensesLabels, ['young', 'myope', 'no', 'reduced'])  
print(classLabel)  # 分类类别：no lenses

