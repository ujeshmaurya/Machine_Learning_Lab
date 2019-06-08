import numpy as np
from matplotlib import pyplot as graphPlot
import random as rn
import math
lamb = 8	#regularisation factor
erry=[]
errx=[]

#function to plot error graph: degree v/s RMSerror
def plotError():
	global erry,errx
	graphPlot.xlabel("Degree of curve")
	graphPlot.ylabel("Error")
	graphPlot.plot(errx,erry)
	graphPlot.show()

#function to calculate error for each degree of polynomial
def findError(equation,j):
	errx.append(j)
	global xd,yd
	i=0
	s=0
	for x in xd:
		y=eval(equation)
		s=s+(y-yd[i])*(y-yd[i])
		i+=1
	ans=math.sqrt(s)/float(len(xd))
	erry.append(ans)

#function to display random generated points on the plane
def showPointsOnPlane():
	global xd,yd
	for i in range(len(xd)):
		graphPlot.scatter(xd[i],yd[i],color = 'green',marker='.')		

#function to show the curve on the plane
def showCurveOnPlane(weightSimple,type):
	global xd,yd,lis
	x = [i for i in range(-1001,1001)]
	color = ""
	if(type==1):			#simple
		color = "blue"
	else:					#regularised
		color = "orange"	
	x = np.array(x)
	equation = str(weightSimple[0][0]) + '+'
	var = "x"
	op = "*"
	for i in range(1,len(weightSimple)):			#find equation from weights
		wt = weightSimple[i][0]
		#equation = equation + str(wt) + '*' + ((str("x")+str("**")+str(i))) + '+'
		equation = equation + str(wt) + '*' + ((var+op)*i)[:-1] + '+'
	equation = equation[:-1]	
	print("Equation of estimated line for degree ",i," : ",equation)
	y = eval(equation)
	findError(equation,i)
	graphPlot.plot(x,y,color = color)
	graphPlot.draw()

#generating random points using random
def generateRandomPoints():
	global xd,yd
	m = rn.uniform(-1,1)
	numPoints = 20
	choose = [-1,1]
	c = 10
	for i in range(numPoints):
		xc = rn.uniform(-800,800)
		yc = m*xc+c
		di =  rn.choice(choose)
		noise = rn.uniform(0,400)
		xd.append(xc)
		yd.append(yc+di*noise)	

#finding the simple and regularised weights using equations
def findWeights(i):
	global xd,yd,lis
	weightSimple = []
	mylist = np.array([[xd[j]**i] for j in range(len(xd))])
	lis = np.append(lis,mylist, axis = 1)
	#take lis as a matrix
	x = np.asmatrix(lis)
	#take yd as a matrix
	y = np.asmatrix(yd)
	#apply the matrix equation Beta = inv((transpose(X)*X))*transpose(X)*Y
	y = np.transpose(y)					
	xt =x.transpose()*x
	b = x.transpose()*y
	c = np.linalg.inv(xt)
	c = c*b
	weightSimple = c.tolist()
	#apply the matrix equation Beta = inv((transpose(X)*X) + lambda*IdentityMAtrix)*transpose(X)*Y
	g = xt + lamb*(np.identity(i+1))
	c = np.linalg.inv(g)
	c = c*b
	regularisedWeights = c.tolist()
	#return both weights
	return weightSimple,regularisedWeights

graphPlot.xlabel("X")
graphPlot.ylabel("Y")
graphPlot.xlim(-1000,1000)
graphPlot.ylim(-1000,1000)
xd=[]
yd=[]
generateRandomPoints()
showPointsOnPlane()
#fill with 1
lis = [[1] for i in range(len((xd)))]
lis = np.array(lis)
np.reshape(lis,(len(xd),1))
#take degree of regression
degree = 9
#find weights for each degree and plot graphs and calculate errors
for i in range(1,degree+1):
	weightSimple, regularisedWeights = findWeights(i)
	print("Simple: ")
	showCurveOnPlane(weightSimple,1)
	print("Regularised: ")
	showCurveOnPlane(regularisedWeights,2)
	graphPlot.pause(1)
	graphPlot.gca().lines[-1].remove()
	graphPlot.gca().lines[-1].remove()
#show the graph for errors
graphPlot.show()
plotError()