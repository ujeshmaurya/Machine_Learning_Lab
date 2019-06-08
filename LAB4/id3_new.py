from csv import *
from collections import Counter
from math import log
#import networkx

from networkx.drawing.nx_agraph import graphviz_layout
import networkx
from matplotlib import pyplot as plt
edge_labels = {}
class Node:
	def __init__(self,data):
		self.data = data
		self.children = {}



def calculateEntropy(classLis):
	#countsList = Counter(dataDic[targetAttrib])
	countsList = Counter(classLis)
	#print(countsList)
	total = sum(countsList.values())
	#print(total)
	entropy = 0
	for i in countsList.values():
		entropy += -i/total * log(i/total,2)
	return entropy	

def getEntropy(targetAttrib,dataDic,dataset):
	global className
	if(targetAttrib == className):
		return calculateEntropy(dataDic[className])
	entropy = 0
	countsList = Counter(dataDic[targetAttrib])
	#print(countsList)
	total = sum(countsList.values())
	# print(total)
	for k,v in countsList.items():
		#print(k,v)
		lis = []
		lis = [dataDic[className][i] for i in range(len(dataDic[className])) if dataDic[targetAttrib][i]==k]
		#print(lis)
		entropy += v/total * calculateEntropy(lis)
	return entropy

def getBestAttrib(dataDic,dataset):
	global className
	targetEntropy = getEntropy(className,dataDic,dataset)
	bestAttrib = ""
	maxIg = -1
	for i in list(dataDic.keys())[1:]:
		entropy = getEntropy(i,dataDic,dataset)
		ig = targetEntropy - entropy
		if(ig>maxIg):
			maxIg = ig
			bestAttrib = i
	# print(targetAttrib)
	return bestAttrib	

def splitDataset(dataset, targetAttrib, attribVal):
	index = dataset[0].index(targetAttrib)
	return [dataset[i][:index]+dataset[i][index+1:] for i in range(len(dataset)) if dataset[i][index]==attribVal or  i==0]

def getDataDic(dataset):
	dataDic = {}	
	for i in range(len(dataset[0])):
		dataDic[dataset[0][i].strip()] = [dataset[j][i] for j in range(1,len(dataset))]
	return dataDic

def pure(dataDic):
	return len(list(set(dataDic[className])))==1


def split(root, dataset, dataDic):
	global className
	if pure(dataDic): 
		return

	bestAttrib = getBestAttrib(dataDic,dataset)
	root.data = bestAttrib
	childList = list(set(dataDic[bestAttrib]))
	# print(bestAttrib)
	for i in childList:	
		root.children[i] = Node(defaultNode)
		newDataset = splitDataset(dataset,bestAttrib,i)
		newDataDic = getDataDic(newDataset)
		if pure(newDataDic):
			root.children[i].data = newDataDic[className][0]
			if (bestAttrib,root.children[i].data) not in edge_labels:
					edge_labels[bestAttrib,root.children[i].data] = i
			#print (root.children[i].data)
		else:
			split(root.children[i],newDataset, newDataDic)
			if (bestAttrib,root.children[i].data) not in edge_labels:
					edge_labels[bestAttrib,root.children[i].data] = i
			#print (root.children[i].data)		

filename = "dataset.csv"
defaultNode = "default"
className = ""
dataset = []
targetEntropy = 0
with open(filename) as csvfile:
	reader  = reader(csvfile)
	dataset = []
	for rows in reader:
		dataset.append(rows)

#print (dataset)
		
for i in range(len(dataset)):
	dataset[i] = [j.strip() for j in dataset[i]]		
dataDic = getDataDic(dataset)
origDataDic = dataDic
origDataset = dataset

#print(origDataDic)
#print(origDataset)
	
className = list(dataDic.keys())[-1]
bestAttrib = getBestAttrib(dataDic,dataset)
#print(bestAttrib)
root = Node(bestAttrib)
childList = list(set(dataDic[bestAttrib]))
print(childList)
for i in childList:
	root.children[i] = Node(defaultNode)
	
	newDataset = splitDataset(dataset,bestAttrib,i)
	newDataDic = getDataDic(newDataset)
	#print (newDataset)
	if pure(newDataDic):
		root.children[i].data = newDataDic[className][0]
		if (bestAttrib,root.children[i].data) not in edge_labels:
				edge_labels[bestAttrib,root.children[i].data] = i
		#print (root.children[i].data,i) 	
	else:
		split(root.children[i],newDataset, newDataDic)
		if (bestAttrib,root.children[i].data) not in edge_labels:
				edge_labels[bestAttrib,root.children[i].data] = i
print (edge_labels)	
		
print("######################################################")
inputFile = open("queries.txt","r")
queries = []
for line in inputFile:
	queries.append(line.split())
queryDic = {}
classes = origDataDic[className]
#print (classes)
for i in range(len(origDataset[0])-2):
	queryDic[origDataset[0][i+1]] = [queries[j][i] for j in range(len(queries))]
#print (queryDic)

for i in range(len(queries)):
	temp = root
	while(temp.data not in classes):
		temp = temp.children[queryDic[temp.data][i]]	
	print("Result class: {} ".format(temp.data))	
graph = networkx.DiGraph()
#edge_labels = {}

for edge in edge_labels:
	graph.add_edges_from([edge])
	#edge_labels[edge[0]] = edge[1]
pos = graphviz_layout(graph,prog='dot')
#graph.add_nodes_from(nodes)
networkx.draw_networkx_edge_labels(graph,pos,edge_labels = edge_labels)
networkx.draw(graph,pos,node_size =1500,with_labels=True,font_weight = 'bold')
#networkx.draw_random(graph)
#networkx.draw_circular(graph)
#networkx.draw_spectral(graph)
plt.show()