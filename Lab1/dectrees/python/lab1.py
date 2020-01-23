import monkdata as m
import dtree as d
import numpy as np
import random
#found on https://stackoverflow.com/questions/9535954/printing-lists-as-tabular-data to print tables
from prettytable import PrettyTable 
import matplotlib.pyplot as plt

#Assignment 0

#Monk 1: a1 and a2 related and it is hard to split on one of the attributes
#Monk 2: True concept has the value of an attribute involved with the value of another attribute and therefore hard to split based on single attribute
#Monk 3: Contains noice and has the smallest set of training data. Alla datasets have small training sets compared to testing sets.

#Assignment 1:

monk = [m.monk1, m.monk2, m.monk3]

entropy_table = PrettyTable(["Dataset", "Entropy"])
	
for i in range(len(monk)):
    row = ["MONK-{0}".format(i+1), round(d.entropy(monk[i]), 10)]
    entropy_table.add_row(row)

print(entropy_table)


#Assignment 2:

#Assignment 3:

#info_gain_table = PrettyTable(["Dataset", "A1", "A2", "A3", "A4", "A5", "A6"])
header = ["Dataset"]
for attr in m.attributes:
	header.append(attr)
info_gain_table = PrettyTable(header)
	

#for i in range(3):
    #row = ["MONK-{0}".format(i+1), round(dt.entropy(monk[i]), 10)]
    #entropy_table.add_row(row)
    
for i in range(len(monk)):
    row = ["MONK-{0}".format(i + 1)]
    for j in range(len(m.attributes)):
      row.append(round(d.averageGain(monk[i], m.attributes[j]), 5))
    info_gain_table.add_row(row)


print(info_gain_table)

#Assignment 4:


#Assignment 5:

monktest = [m.monk1test, m.monk2test, m.monk3test]

error_table = PrettyTable(['Dataset', 'Error_Train', 'Error_Test'])

for i in range(len(monk)):
    row = ["MONK-{0}".format(i + 1)]
    t = d.buildTree(monk[i], m.attributes)
    row.append(round(1 - d.check(t, monk[i]), 5))
    row.append(round(1 - d.check(t, monktest[i]), 5))

    error_table.add_row(row)

print(error_table)


#Assignment 6:



#Assignment 7:

def partition(data, fraction):
	ldata = list(data)
	random.shuffle(ldata)
	breakPoint = int(len(ldata) * fraction)
	return ldata[:breakPoint], ldata[breakPoint:]

monk1train, monk1val = partition(m.monk1, 0.6)

#check function Measure fraction of correctly classified samples
#The error is therefore 1-correct


fraction = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]



def prune(data, test):
	pruned_trees_fraction = []
	for frac in fraction:
		train, val = partition(data, frac)
		t = d.buildTree(train, m.attributes)
		all_pruned = d.allPruned(t)
		
		#I suppose t is included in all_pruned
		best_tree_perf = 0
		for t in all_pruned:
			candidate_perf = d.check(t,val)
			if best_tree_perf < candidate_perf:
				best_tree_perf = candidate_perf
				best_tree = t
		
		pruned_trees_fraction.append(1-d.check(best_tree, test))
	return pruned_trees_fraction



monk1_error_prune = []
monk3_error_prune = []

for i in range(1000):
	monk1_error_prune.append(prune(m.monk1, m.monk1test))
	monk3_error_prune.append(prune(m.monk3, m.monk3test))


monk1_mean_error = np.mean(monk1_error_prune, axis = 0)
monk3_mean_error = np.mean(monk3_error_prune, axis = 0)
monk1_sd_error = np.std(monk1_error_prune, axis = 0)
monk3_sd_error = np.std(monk3_error_prune, axis = 0)



print(monk1_mean_error)

print(monk3_mean_error)

print(monk1_sd_error)

print(monk3_sd_error)



#################PLOTS

#plot only the mean error vs fractions
plt.plot(fraction, monk1_mean_error,label="Monk-1", marker='o')
plt.plot(fraction, monk3_mean_error,label="Monk-3", marker='o')
plt.legend(loc="upper right")
plt.xlim(0, 1) 
plt.title("Mean Error vs Partition Fraction")
plt.xlabel("Fraction")
plt.ylabel("Mean Error")
plt.show()


#plot the mean error vs fractions and standard deviation
plt.plot(fraction, monk1_mean_error,label="Monk-1", marker='o', color='blue')
plt.plot(fraction, monk3_mean_error,label="Monk-3", marker='o', color='red')
plt.plot(fraction, monk1_sd_error, marker='o', color='blue')
plt.plot(fraction, monk3_sd_error, marker='o', color='red')
plt.legend(loc="upper right")
plt.xlim(0, 1) 
plt.title("Standard Error vs Partition Fraction")
plt.xlabel("Fraction")
plt.ylabel("Mean Error")
rectangle = plt.Rectangle((0.25,0.027), 0.6, 0.03, fc="#f2f2f2", ec="black", linestyle = 'dashed')
plt.gca().add_patch(rectangle)
plt.text(0.35, 0.06, "Standard Deviation of Error")
plt.show()



#plot the mean error vs fractions and standard deviation version 2
plt.plot(fraction, monk1_mean_error,label="Monk-1", marker='o')
plt.plot(fraction, monk3_mean_error,label="Monk-3", marker='o')
plt.legend(loc="upper right")
plt.xlim(0, 1) 
plt.title("Mean Error vs Partition Fraction")
plt.xlabel("Fraction")
plt.ylabel("Mean Error")

a = plt.axes([0.2, 0.2, 0.2, 0.2])
plt.plot(fraction, monk1_sd_error,label="Monk-1 Sd", marker='o')
plt.plot(fraction, monk3_sd_error,label="Monk-3 Sd", marker='o')
plt.title('Standard Error')
plt.xlim(0, 1)
plt.ylim(0, 0.2)
plt.yticks([0.00, 0.10, 0.20])
plt.xticks(rotation=45)
#plt.yticks([])
plt.show()


