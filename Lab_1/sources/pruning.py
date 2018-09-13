import random
import dtree as dtree
import monkdata as m
# import drawtree_qt5 as drawtree

def partition(data, fraction) :
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

def pruneTree(tree, oldError, validationTree) :
    pruningErrors = list()

    prunedTrees = dtree.allPruned(tree)

    for trees in prunedTrees :
        pruningErrors.append(dtree.check(trees, validationTree))

    bestError = max(pruningErrors)
    indexOfBestError = pruningErrors.index(bestError)

    if bestError < oldError :
        return tree, oldError
    else:
        return pruneTree(prunedTrees[indexOfBestError], bestError, validationTree)


###
# Initialize lists of treesto be pruned

fractions = list((0.3, 0.4, 0.5, 0.6, 0.7, 0.8))

monk1Trees = list()
for i in range(0, len(fractions)) :
    monk1Trees.append(list())

monk1ValTrees = list()
for i in range(len(fractions)) :
    monk1ValTrees.append(list())

monk3Trees = list()
for i in range(len(fractions)) :
    monk3Trees.append(list())

monk3ValTrees = list()
for i in range(len(fractions)) :
    monk3ValTrees.append(list())

# Randomize training and validation sets

for i in range(len(fractions)) :
    for j in range(128) :
        monk1train, monk1val = partition(m.monk1, fractions[i])
        monk3train, monk3val = partition(m.monk3, fractions[i])

        monk1Trees[i].append(dtree.buildTree(monk1train, m.attributes))
        monk1ValTrees[i].append(monk1val)

        monk3Trees[i].append(dtree.buildTree(monk3train, m.attributes))
        monk3ValTrees[i].append(monk3val)

###
# Write data to file: open file

txt = open("data.txt", "w")

txt.write("monk1\n")
for i in range(len(monk1Trees)) :
    txt.write(str("\n\n" + "Fraction: " + str(fractions[i]) + "\n\n"))
    for j in range(len(monk1Trees[i])) :
        prunedTree, prunedError = pruneTree(monk1Trees[i][j], dtree.check(monk1Trees[i][j], monk1ValTrees[i][j]), monk1ValTrees[i][j])
        txt.write(str(prunedError) + "\n")

print("Finished monk1")

txt.write("\n")
txt.write("monk3\n")
for i in range(len(monk3Trees)) :
    txt.write("\n\n" + "Fraction: " + str(fractions[i]) + "\n\n")
    for j in range(len(monk3Trees[i])) :
        prunedTree, prunedError = pruneTree(monk3Trees[i][j], dtree.check(monk3Trees[i][j], monk3ValTrees[i][j]), monk3ValTrees[i][j])
        txt.write(str(prunedError) + "\n")

print("Finished monk3")