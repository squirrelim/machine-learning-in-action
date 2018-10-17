import trees
import treePlotter

# myDat,labels=trees.createDataSet()
# print(myDat)
# Ent=trees.calShannonEnt(myDat)
# print(Ent)

# trees.splitDataSet(myDat,0,1)
# myDat,labels=trees.createDataSet()

# bestFeature=trees.chooseBestFeatureToSplit(myDat)
# print(bestFeature)

# myTree=trees.createTree(myDat,labels)
# print(myTree)

# treePlotter.createPlot()
# mytree=treePlotter.retrieveTree(0)
# numLeafs=treePlotter.getNumLeafs(mytree)
# print(numLeafs)
# treeDepth=treePlotter.getTreeDepth(mytree)
# print(treeDepth)

# myTree=treePlotter.retrieveTree(0)
# treePlotter.createPlot(myTree)
# print(myTree)
# myTree['no surfacing'][3]='maybe'
# print(myTree)
# treePlotter.createPlot(myTree)

myDat,labels=trees.createDataSet()
myTree=treePlotter.retrieveTree(0)
res=trees.classify(myTree,labels,[1,1])
print(res)
print(myTree)
trees.storeTree(myTree,'classifierStorage.txt')
newTree=trees.grabTree('classifierStorage.txt')
print(newTree)

fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = trees.createTree(lenses, lensesLabels)
treePlotter.createPlot(lensesTree)