'''
Created on Oct 27, 2017

kNN: k Nearest Neighbors

@author: mabing
'''

from numpy import *
import operator
from os import listdir

'''
使用 kNN 进行分类（约会网站配对）

@param inX：用于分类的数据向量
@param dataSet：训练样本集
@param labels：标签向量
@param k：用于选择最近邻的数目
'''
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # 求inX与数据集中各个样本的欧氏距离
    diffMat = tile(inX, (dataSetSize,1)) - dataSet   # numpy中的tile函数将inX复制为重复的dataSize个行和重复的1列，功能相当于MATLAB中的repmat
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)  # 按照x轴相加
    distances = sqDistances**0.5  
    sortedDistIndicies = distances.argsort()   # 从小到大排序后，返回索引
    # 字典，key存储第i小的标签值，value为标签的次数
    classCount = {}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]  # 取第i个小的标签值
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1  # 根据标签统计标签次数，如果没找到返回0。统计前k个候选者中标签出现的次数
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) # operator.itemgetter(1) 按照第2个元素，即标签出现的次数对classCount从大到小排序
    # print(sortedClassCount)  # 测试结果 [('B', 2), ('A', 1)]
    return sortedClassCount[0][0]  # 返回最终的类别，即标签值key

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

'''
# 测试kNN
group, labels = createDataSet()
ans = classify0([0,0], group, labels, 3)
print(ans)
'''

'''
读取文本记录，提取特征矩阵和标签向量
@param filename：文件名（如 datingTestSet2.txt）
'''
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         # 得到文件行数
    returnMat = zeros((numberOfLines,3))        # 构造一个numberOfLines行3列的全0矩阵，3列代表3个特征，用来存储特征
    classLabelVector = []                       # 存储标签向量  
    fr = open(filename)
    index = 0
    for line in fr.readlines():  # 读文件时，每一行都是一个字符串，故line就是一个字符串
        line = line.strip()  # 去掉一行字符串line的前后空格
        listFromLine = line.split('\t')  # 以 '\t' 为切片，切成List，存储各个特征和最后一行的标签值
        returnMat[index,:] = listFromLine[0:3]  # 存储特征
        classLabelVector.append(int(listFromLine[-1]))  # 存储标签值
        index += 1
    return returnMat,classLabelVector

'''
# 使用Matplotlib绘制原始数据的散点图

datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure('figure1')
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1], datingDataMat[:,2])  # 绘制散点图，注意，是第2列，第3列数据
plt.show() 

fig = plt.figure('figure2')
ax = fig.add_subplot(111)
# 利用变量datingLabels存储的类标签属性，在图上绘制色彩不等，尺寸不同的点
ax.scatter(datingDataMat[:,0], datingDataMat[:,1], 15.0*array(datingLabels), 15.0*array(datingLabels)) 
plt.show()
'''


'''
归一化数值
@param dataSet：训练样本集
'''
def autoNorm(dataSet):
    minVals = dataSet.min(0)  # 0代表从列中取最小值
    maxVals = dataSet.max(0)  
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet)) # 构造一个和dataSet一样大小的矩阵
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet / tile(ranges, (m,1))   # 特征值相除，得到正则化后的新值
    return normDataSet, ranges, minVals

'''
# 测试归一化数值
datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
normMat, ranges, minVals = autoNorm(datingDataMat)
print(normMat)
print(ranges)
print(minVals)
'''

def datingClassTest():
    hoRatio = 0.08      # 随机挖去 10% 的数据作为测试集
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       # 加载数据文件
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)  # 随机挖去的行数
    errorCount = 0.0 
    for i in range(numTestVecs):
    	# 前numTestVecs条作为测试集（一个一个测试），后面的数据作为训练样本，训练样本的标签，3个近邻
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("The number of errr is: %d" % int(errorCount))
    print("The total error rate is: %f" % (errorCount / float(numTestVecs)))

'''
# 测试分类错误率，错误分类个数
datingClassTest()  
'''

'''
使用 kNN 进行分类（手写识别系统）
@param filename：文件名
'''
def img2vector(filename):
    returnVect = zeros((1,1024))  # 图像大小为32*32，将其转化为1*1024的向量
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []  # 保存数字类别
    trainingFileList = listdir('digits/trainingDigits')  # 返回指定的文件夹包含的文件或文件夹的名字的列表，这个列表以字母顺序
    m = len(trainingFileList)  # 文件数量
    trainingMat = zeros((m,1024))  # 训练矩阵，m行训练数据，每一行1024个图像特征
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]   # 去掉后缀.txt
        classNumStr = int(fileStr.split('_')[0])  # 得到数字类别
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('digits/trainingDigits/%s' % fileNameStr)
    # 测试集
    testFileList = listdir('digits/testDigits')        
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     # 去掉后缀.txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        # 使用 kNN 分类
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("the total error rate is: %f" % (errorCount / float(mTest)))

# 手写体识别测试
handwritingClassTest()

