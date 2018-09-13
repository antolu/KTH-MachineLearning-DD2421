import dtree as dtree
import monkdata as m
# import drawtree_qt5 as drawtree 

tree1 = dtree.buildTree(m.monk1, m.attributes)
print("Training data monk1", dtree.check(tree1, m.monk1))
print("Testing data monk1", dtree.check(tree1, m.monk1test))

tree2 = dtree.buildTree(m.monk2, m.attributes)
print("Training data monk2", dtree.check(tree2, m.monk2))
print("Testing data monk2", dtree.check(tree2, m.monk2test))

tree3 = dtree.buildTree(m.monk3, m.attributes)
print("Training data monk3", dtree.check(tree3, m.monk3))
print("Testing data monk3", dtree.check(tree3, m.monk3test))