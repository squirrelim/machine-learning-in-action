import bayes
from numpy import *
listOPosts,listClasses=bayes.loadDataSet()
# print(listOPosts)
myVocalList=bayes.createVocabList(listOPosts)
# print(myVocalList)
#
# print(bayes.setOfwords2Vec(myVocalList,listOPosts[0]))
# trainMat=[]
# for postinDoc in listOPosts:
#     trainMat.append(bayes.setOfwords2Vec(myVocalList,postinDoc))
# p0V,p1V,pAb=bayes.trainNB0(trainMat,listClasses)
# print(pAb,p0V,p1V)
print(bayes.testingNB())