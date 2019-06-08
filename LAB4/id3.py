from csv import *
from collections import Counter
from math import log
# Self referencial class for creating decision tree
class Node:
	def __init__(self,data):
		self.data = data 			#Stores a string which is the class name
		self.children = {}			#Stores a link to the next node of same type

# Function to calculate Entropy
def calculateEntropy(classLis):
	countsList = Counter(classLis)			#Calculate count of each data in the feature
	total = sum(countsList.values())		#Count of all data in the current feature
	entropy = 0								#set initial entropy to zero
	for i in countsList.values():			#iterate for all distinct data in that feature
		entropy += -i/total * log(i/total,2)#Find entropy using probabilities of each data
	return entropy							#return Final entropy value

# Function to calculate entropy of all features w.r.t. a dataset and targetAttribute
def getEntropy(targetAttrib,dataDic,dataset):
	global className						#initially it will be vegetation
	if(targetAttrib == className):			#simply return the entropy of target attribute
		return calculateEntropy(dataDic[className])
	entropy = 0
	countsList = Counter(dataDic[targetAttrib])	#Calculate count of each data in the feature
	total = sum(countsList.values())			#Count of all attributes in the current feature
	for k,v in countsList.items():				#'k' is the attribute name, 'v' is it's count
		lis = []
		lis = [dataDic[className][i] for i in range(len(dataDic[className])) if dataDic[targetAttrib][i]==k]
		entropy += v/total * calculateEntropy(lis)
	return entropy

def getBestAttrib(dataDic,dataset):
	global className
	targetEntropy = getEntropy(className,dataDic,dataset)
	bestAttrib = ""
	maxIg = -1
	for i in list(dataDic.keys())[1:]:		#iterate through all features
		entropy = getEntropy(i,dataDic,dataset)		#find it's entropy
		ig = targetEntropy - entropy 				#get info gain
		if(ig>maxIg):								#save highest info gain and attribute
			maxIg = ig
			bestAttrib = i
	return bestAttrib								#return best attribute

# Split the dataset according to an attribute value
def splitDataset(dataset, targetAttrib, attribVal):
	index = dataset[0].index(targetAttrib)	
	return [dataset[i][:index]+dataset[i][index+1:] for i in range(len(dataset)) if dataset[i][index]==attribVal or  i==0]

# make a datadictionary from the given list of dataset
def getDataDic(dataset):
	dataDic = {}	
	for i in range(len(dataset[0])):
		dataDic[dataset[0][i].strip()] = [dataset[j][i] for j in range(1,len(dataset))]
	return dataDic

# returns true if the dataDic[className] contains one distinct value, to identify termination of a tree
def pure(dataDic):
	return len(list(set(dataDic[className])))==1

# recursive function to build a decision tree
def split(root, dataset, dataDic, bestAttribute):
	global graph
	global className
	if pure(dataDic): 				# termination node of tree
		return
	bestAttrib = getBestAttrib(dataDic,dataset)	#select the best attribute
	root.data = bestAttrib 						#define the root of tree as best attribute
	childList = list(set(dataDic[bestAttrib]))  #define the child features of the best attribute
	for i in childList:	
		root.children[i] = Node(defaultNode)
		newDataset = splitDataset(dataset,bestAttrib,i)	# split w.r.t. i'th feature
		newDataDic = getDataDic(newDataset)
		if pure(newDataDic):							# termination of tree
			root.children[i].data = newDataDic[className][0]
			graph[(bestAttrib, root.children[i].data)] = i
		else:											#recursively build tree further
			#graph[(bestAttrib, root.children[i].data)] = i
			split(root.children[i],newDataset, newDataDic,bestAttrib)		

filename = "dataset.csv"
defaultNode = "default"
graph = {}
className = ""
dataset = []
targetEntropy = 0
with open(filename) as csvfile:
	reader  = reader(csvfile)
	dataset = []
	for rows in reader:
		dataset.append(rows)
		
for i in range(len(dataset)):
	dataset[i] = [j.strip() for j in dataset[i]]		
dataDic = getDataDic(dataset)
origDataDic = dataDic
origDataset = dataset
className = list(dataDic.keys())[-1]
bestAttrib = getBestAttrib(dataDic,dataset)
root = Node(bestAttrib)
childList = list(set(dataDic[bestAttrib]))
for i in childList:
	root.children[i] = Node(defaultNode)
	newDataset = splitDataset(dataset,bestAttrib,i)
	newDataDic = getDataDic(newDataset)
	if pure(newDataDic):
		root.children[i].data = newDataDic[className][0]
		graph[(bestAttrib, root.children[i].data)] = i
	else:
		#graph[(bestAttrib, root.children[i].data)] = i
		split(root.children[i],newDataset, newDataDic, bestAttrib)	
print(graph)
print("****************TESTING*********************")
inputFile = open("queries.txt","r")
queries = []
for line in inputFile:
	queries.append(line.split())
queryDic = {}
classes = origDataDic[className]
for i in range(len(origDataset[0])-2):
	queryDic[origDataset[0][i+1]] = [queries[j][i] for j in range(len(queries))]

for i in range(len(queries)):
	temp = root
	while(temp.data not in classes):
		temp = temp.children[queryDic[temp.data][i]]	
	print("Result class: {} ".format(temp.data))	


