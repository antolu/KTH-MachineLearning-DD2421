import monkdata as m
import dtree as dtree

print("monk1: ")
print("a1", dtree.averageGain(m.monk1, m.attributes[0]), 
        "\na2", dtree.averageGain(m.monk1, m.attributes[1]), 
        "\na3", dtree.averageGain(m.monk1, m.attributes[2]), 
        "\na4", dtree.averageGain(m.monk1, m.attributes[3]), 
        "\na5", dtree.averageGain(m.monk1, m.attributes[4]),
        "\na6", dtree.averageGain(m.monk1, m.attributes[5]))
print("Best attribute: ", dtree.bestAttribute(m.monk1, m.attributes), "\n")


print("monk2: ")
print("a1", dtree.averageGain(m.monk2, m.attributes[0]), 
        "\na2", dtree.averageGain(m.monk2, m.attributes[1]), 
        "\na3", dtree.averageGain(m.monk2, m.attributes[2]), 
        "\na4", dtree.averageGain(m.monk2, m.attributes[3]), 
        "\na5", dtree.averageGain(m.monk2, m.attributes[4]),
        "\na6", dtree.averageGain(m.monk2, m.attributes[5]))
print("Best attribute: ", dtree.bestAttribute(m.monk2, m.attributes), "\n")

print("monk3: ")
print("a1", dtree.averageGain(m.monk3, m.attributes[0]), 
        "\na2", dtree.averageGain(m.monk3, m.attributes[1]), 
        "\na3", dtree.averageGain(m.monk3, m.attributes[2]), 
        "\na4", dtree.averageGain(m.monk3, m.attributes[3]), 
        "\na5", dtree.averageGain(m.monk3, m.attributes[4]),
        "\na6", dtree.averageGain(m.monk3, m.attributes[5]))
print("Best attribute: ", dtree.bestAttribute(m.monk3, m.attributes), "\n")