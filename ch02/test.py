import kNN
group, labels = kNN.createDataSet()
print(group)

res=kNN.classify0([0,0],group,labels,3)
print(res)

datingDataMat,datingLabels=kNN.file2matrix('datingTestSet2.txt')
# print(datingDataMat)
# print(datingLabels[0:20])

# kNN.analysisdata(datingDataMat,datingLabels)
# kNN.datingClassTest()
# kNN.classifyPerson()

#手写识别
# testVector=kNN.img2vector('testDigits/0_13.txt')
# print(testVector[0,32:63])

kNN.handwritingClassTest()