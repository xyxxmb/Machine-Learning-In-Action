'''
Created on Nov 25, 2017

@author: mabing
'''
import sys
from numpy import mat, mean, power

def read_input(file):
    for line in file:
        yield line.rstrip()
       
input = read_input(sys.stdin) # 读取mapper的输出，以list返回

# 将mapper中的3个输出用'\t'划分，组成一个列表存入mapperOut
mapperOut = [line.split('\t') for line in input]  # input等价于["xx\tyy\t"]，list过滤后仍然是一个list整体，即 [[xx,yy]]，而不是[[xx],[yy]]
# 注意，mapperOut可能有来自不同机器的多个list，最终的形式可能为[[m1],[m2],[m3],[...]]，mi为第i个机器产生的mapper输出

cumVal=0.0
cumSumSq=0.0
cumN=0.0
for instance in mapperOut:  # 对于每一个mapper的输出
    nj = float(instance[0])
    cumN += nj  # 浮点数数目总和 
    cumVal += nj*float(instance[1])  # 均值总和
    cumSumSq += nj*float(instance[2]) # 平方后均值总和
    
# 计算全局的均值和全局的平方后均值
mean = cumVal/cumN
meanSq = cumSumSq/cumN

#output size, mean, mean(square values)
print("%d\t%f\t%f" % (cumN, mean, meanSq))
print("report: still alive", file=sys.stderr)

# windows下执行命令:
# python mrMeanMapper.py < inputFile.txt | python mrMeanReducer.py
# inputFile.txt 每一行为"浮点数\n"