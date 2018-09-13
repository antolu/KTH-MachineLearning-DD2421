import dtree as dtree
import monkdata as m
import drawtree_qt5 as drawtree

# firstLeaf = dtree.select(m.monk1, m.attributes[4], 1)
# print("a1", dtree.averageGain(firstLeaf, m.attributes[0]), 
#         "\na2", dtree.averageGain(firstLeaf, m.attributes[1]), 
#         "\na3", dtree.averageGain(firstLeaf, m.attributes[2]), 
#         "\na4", dtree.averageGain(firstLeaf, m.attributes[3]), 
#         "\na5", dtree.averageGain(firstLeaf, m.attributes[4]),
#         "\na6", dtree.averageGain(firstLeaf, m.attributes[5]))
# print("Best attribute: ", dtree.bestAttribute(firstLeaf, m.attributes), "\n")

# secondLeaf = dtree.select(m.monk1, m.attributes[4], 2)
# print("a1", dtree.averageGain(secondLeaf, m.attributes[0]), 
#         "\na2", dtree.averageGain(secondLeaf, m.attributes[1]), 
#         "\na3", dtree.averageGain(secondLeaf, m.attributes[2]), 
#         "\na4", dtree.averageGain(secondLeaf, m.attributes[3]), 
#         "\na5", dtree.averageGain(secondLeaf, m.attributes[4]),
#         "\na6", dtree.averageGain(secondLeaf, m.attributes[5]))
# print("Best attribute: ", dtree.bestAttribute(secondLeaf, m.attributes), "\n")

# thirdLeaf = dtree.select(m.monk1, m.attributes[4], 3)
# print("a1", dtree.averageGain(thirdLeaf, m.attributes[0]), 
#         "\na2", dtree.averageGain(thirdLeaf, m.attributes[1]), 
#         "\na3", dtree.averageGain(thirdLeaf, m.attributes[2]), 
#         "\na4", dtree.averageGain(thirdLeaf, m.attributes[3]), 
#         "\na5", dtree.averageGain(thirdLeaf, m.attributes[4]),
#         "\na6", dtree.averageGain(thirdLeaf, m.attributes[5]))
# print("Best attribute: ", dtree.bestAttribute(thirdLeaf, m.attributes), "\n")

# fourthLeaf = dtree.select(m.monk1, m.attributes[4], 4)
# print("a1", dtree.averageGain(fourthLeaf, m.attributes[0]), 
#         "\na2", dtree.averageGain(fourthLeaf, m.attributes[1]), 
#         "\na3", dtree.averageGain(fourthLeaf, m.attributes[2]), 
#         "\na4", dtree.averageGain(fourthLeaf, m.attributes[3]), 
#         "\na5", dtree.averageGain(fourthLeaf, m.attributes[4]),
#         "\na6", dtree.averageGain(fourthLeaf, m.attributes[5]))
# print("Best attribute: ", dtree.bestAttribute(fourthLeaf, m.attributes), "\n")

tree = dtree.buildTree(m.monk1, m.attributes, 2)

print(tree)

drawtree.drawTree(tree)
