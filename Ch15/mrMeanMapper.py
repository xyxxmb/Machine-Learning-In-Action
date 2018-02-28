'''
Created on Nov 25, 2017

Machine Learning in Action Chapter 15
Map Reduce Job for Hadoop Streaming 

mrMeanMapper.py

@author: mabing
'''
import sys
from numpy import mat, mean, power

def read_input(file):
    for line in file:
        yield line.rstrip()  # line.rstrip()，去掉右边空格或回车，只留下浮点数，每一次返回一个list
# yield把函数read_input()看成一个generator，调用该函数执行到yield返回一个iterable对象。下次下次迭代时，代码从yield的下一条语句继续执行
        
input = [float(line) for line in read_input(sys.stdin)] # 从文件中读取每一行数据
numInputs = len(input)
input = mat(input)
sqInput = power(input,2)

# output size, mean, mean(square values)
print("%d\t%f\t%f" % (numInputs, mean(input), mean(sqInput))) # 计算每一列的平均值和平方后的均值，结果为 100     0.509570        0.344439
print("report: still alive", file=sys.stderr)  # file=sys.stderr 向标准错误流输出发送报告

# windows下执行命令:
# python mrMeanMapper.py < inputFile.txt
# inputFile.txt 每一行为"浮点数\n"
