'''
Created on Nov 20, 2017

Ch 11 Apriori

@author: mabing
'''
from numpy import *

# 用于测试的简单数据集，每一个lis代表一次购买的组合，如第一次购买记录为购买商品 1,3,4
def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

# 创建大小为1的所有候选项集的集合C1
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:    # 对于每条购买记录
        for item in transaction:   # 对于每件商品
            if not [item] in C1:   # 如果该商品还没有加入C1集合，这里item加上[]的原因是在C1中存放的也是[item]，所有要判断与[item]不重复的项
                C1.append([item])  # C1只包含大小为1的项集，对于测试数据为 [[1],[2],[3],[4],[5]]
                                   # 这里为每项创建一个列表的原因：①Python不支持一个整数变为集合，即set(1)是不可迭代的。②后续需要做集合操作
    C1.sort()  # 排序
    return list(map(frozenset, C1)) # 对C1中每项构建一个不变集合，用frozenset类型，表示不可改变的集合，即用户不能修改它们，而set不能满足此要求

'''
计算满足最小支持度要求的那些项集以及每个项集的支持度
@param D：数据集，已将每一个购买记录转化为一个集合
@param Ck：候选项集列表
@param minSupport：感兴趣项集的最小支持度
'''                
def scanD(D, Ck, minSupport):
    ssCnt = {}  # key为Ck集合，value为Ck在D中出现的次数
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not can in ssCnt: ssCnt[can]=1 # 如果将frozenset改为set，会报错：set是unhashable。因为list、set、dict等是可变类型，不能hash
                else: ssCnt[can] += 1
    # print(ssCnt)  # 对于测试数据 {frozenset({1}): 2, frozenset({3}): 3, frozenset({4}): 1, frozenset({2}): 3, frozenset({5}): 3}
    numItems = float(len(D))  # 集合中记录的个数
    retList = [] 
    supportData = {}   # key为Ck集合，value为项集的支持度
    for key in ssCnt:
        support = ssCnt[key] / numItems # 计算支持度=项集出现次数/总记录数
        if support >= minSupport:
            retList.insert(0,key)  # 保留满足最小支持度要求的那些项集
        supportData[key] = support
    return retList, supportData  # 返回满足最小支持度要求的那些项集、每个项集的支持度

'''
# 测试Apriori算法的辅助函数
dataSet = loadDataSet()
C1 = createC1(dataSet)  # [frozenset({1}), frozenset({2}), frozenset({3}), frozenset({4}), frozenset({5})]
D = list(map(set, dataSet)) # 将数据集转化为集合
L1, suppData0 = scanD(D,C1,0.5)
print(L1)  # [frozenset({5}), frozenset({2}), frozenset({3}), frozenset({1})]
print(suppData0)  # {frozenset({1}): 0.5, frozenset({3}): 0.75, frozenset({4}): 0.25, frozenset({2}): 0.75, frozenset({5}): 0.75}
'''

'''
生成项集函数：如 输入为 [{0}, {1}, {2}]，输出为 [{0,1}, {0,2}, {1,2}]
@param Lk：旧项集元素列表，如 [{0}, {1}, {2}]
@param k：新项集中元素的个数，如2
'''
def aprioriGen(Lk, k): 
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk): 
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]  # L1，L2目的是防止出现重复的集合，减少不必要的迭代次数
            L1.sort(); L2.sort()
            if L1 == L2: # 如果前k-2个元素相同
                retList.append(Lk[i] | Lk[j]) # 则使用并操作将两个集合合并成一个大小为k的集合
    return retList  # 新项集组成的列表，每个元素还是一个frozenset集合

'''
Apriori算法：产生满足最小支持度的所有频繁项集
@param dataSet：每一条购买记录组成的原始lsit数据集
@param minSupport：最小支持度
'''
def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)  
    D = list(map(set, dataSet)) 
    L1, supportData = scanD(D, C1, minSupport)  # 计算满足最小支持度要求的那些项集以及每个项集的支持度
    L = [L1]  
    k = 2 
    while (len(L[k-2]) > 0):  # 因为遍历完所有项集后，L列表最后一个一定为[]，所以以此作为终止条件
        Ck = aprioriGen(L[k-2], k)  # 生成新项集列表
        # print(Ck)
        Lk, supK = scanD(D, Ck, minSupport)  # 用新项集继续计算满足最小支持度要求的那些项集以及每个项集的支持度
        supportData.update(supK)  # 在原来的项集的基础上添加每个新项集的支持度
        # print(supportData)
        L.append(Lk) 
        k += 1
    return L, supportData # 返回满足要求的所有项集、包含这些项集的支持度

# 测试Apriori算法
dataSet = loadDataSet()
L, supportData = apriori(dataSet)
print(L) 
'''
结果：注意，最后一个返回[]
[[frozenset({5}), frozenset({2}), frozenset({3}), frozenset({1})], [frozenset({2, 3}), frozenset({3, 5}), frozenset
({2, 5}), frozenset({1, 3})], [frozenset({2, 3, 5})], []]
'''
print(supportData)
'''
结果：第一次，因为frozenset({4}): 0.25<0.5，所以会被剔除，故当项集元素大于1的时候，就不再出现所有4的集合了，同理其他也是一样
{frozenset({1}): 0.5, frozenset({3}): 0.75, frozenset({4}): 0.25, frozenset({2}): 0.75, frozenset({5}): 0.75, froze
nset({1, 3}): 0.5, frozenset({2, 5}): 0.75, frozenset({3, 5}): 0.5, frozenset({2, 3}): 0.5, frozenset({1, 5}): 0.25
, frozenset({1, 2}): 0.25, frozenset({2, 3, 5}): 0.5}
'''

'''
关联规则生成函数：产生满足最小可信(置信)度阈值的规则
@param L：频繁项集列表
@param supportData：包含频繁项集支持数据的字典（其中有些项集是不支持的，如上面supportData中的输出结果，有些支持度小于0.5的项集就不属于）
@param minConf：最小可信(置信)度阈值
'''
def generateRules(L, supportData, minConf=0.7):  
    bigRuleList = []  # 存储所有关联规则的列表
    for i in range(1, len(L)): # 从项集中元素大于1个的项集开始，因为无法从单元素项集中构建关联规则
        for freqSet in L[i]:
            # 如果从[{0,1,2}]开始，则H1= [{0},{1},{2}]。使用[item]是因为set不支持整数迭代，故先将整数存储为一个个list，再转化为frozenset
            H1 = [frozenset([item]) for item in freqSet]  
            # print(H1)
            if (i > 1): # 如果项集中大于两个元素，要做进一步合并
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:   # 如果项集中只有两个元素，使用calcConf()函数来计算可信度
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList         

'''
计算可信度
@param freqSet：一个项集
@param H：该项集划分成的包含单个元素的元素集合的列表
@param supportData：包含频繁项集支持数据的字典
@param brl：存储所有关联规则的列表
@param minConf：可信度，默认0.7
'''
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = [] # 保存该项集产生的关联规则的右侧
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq] # 如，计算关联规则0->1的可信度公式={0,1}/{0}
        if conf >= minConf: 
            print(freqSet-conseq,'-->',conseq,'conf:',conf) # 输出关联规则和对应的可信度
            brl.append((freqSet-conseq, conseq, conf)) # 存储关联规则，这里为传引用
            prunedH.append(conseq)
    return prunedH # 返回该项集产生的关联规则的右侧

'''
# 从最初的项集中生成更多的关联规则，如 {2}->{3},{2}->{5},来产生{2}->{3,5}这条新规则
@param H：关联规则右侧的元素列表H
其他参数与calcConf()中一致
'''
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)):  # 如果项集元素数目大于右侧元素+1，则满足进一步合并右侧的条件
        Hmp1 = aprioriGen(H, m+1) # 生成项集元素为m+1个的新项集，如 [{2},{3},{5}]变成[{2,3},{2,5},{3,5}]
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)  # 将Hmp1作为关联规则的右侧，计算可信度，返回关联规则的右侧
        if (len(Hmp1) > 1):       # 如果不止一条规则满足，则看看是否可以进一步组合这些规则
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

# 测试关联规则生成函数
print()
dataSet = loadDataSet()
L, supportData = apriori(dataSet, minSupport=0.5) # 支持度为0.5
rules = generateRules(L, supportData, minConf=0.5) #　可信度为0.5，如果改得大一些，则关联规则较少
print(rules)

# 发现毒蘑菇的相似特征
mushDataSet = [line.split() for line in open('mushroom.dat').readlines()] # 每一行是字符串，line.split()按照空格划分，每行组成一个list
# print(mushDataSet[:2])  # 打印前两个list
L, suppData = apriori(mushDataSet, minSupport=0.3)  
print()
for item in L[1]:  # 项集中含有两个特征
    if item.intersection('2'):  # 如果项集中包含'2'
        print(item)  # frozenset({'28', '2'})、frozenset({'53', '2'})...
for item in L[3]:    # 项集中含有4个特征
    if item.intersection('2'):  # 如果项集中包含特征'2'
        print(item)  # frozenset({'28', '59', '2', '34'})、frozenset({'28', '63', '2', '34'})
# 意味着如果2表示有毒，那么和2有关的数字都不要尝试