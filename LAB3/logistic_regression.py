import numpy as np
from matplotlib import pyplot as plt
import random as rn
import math
from pylab import *

#here W is actually transpose of W
#aim is to return theta(y.W.X)
def sigmoid(y,w,x) :
    temp = x*w
    s = -1*y*temp
    return 1/(1+math.exp(s))

#aim is to plot the points '1' and '4' on the graph using coordinates as its features
#blue is 1 and red is 4
def draw_point(x1,x2,y):
    for i in range(len(x1)):
        if(y==1) :
            scatter(x1[i],x2[i],color = 'blue',marker='.')
        else :
            scatter(x1[i],x2[i],color = 'red',marker='.')

#remove the last plotted line and make a new line for a better iteration of gradient descent
#make curve for values of x = {0.80,0.81,...,1.00}; calculate 'y' using equation and plot
def show_curve(w) :
    w = np.array(w)
    x = [i/100 for i in range (80,101)]
    x = np.array(x)
    equation = '-' + str(w[0][0]/w[0][2])+ '-' + str(w[0][1]/w[0][2]) + "*x"
    y = eval(equation)
    plot(x,y,color='magenta')

#read from input file
input_file = open("test.txt","r")
data = input_file.readlines()
input_file.close()
#make list for storing features
#feature 'a' is average intensity of image
#feature 'b' is average of absolute difference of mirror image intensity in image
i1 = []		#feature 'a' for digit 1
i4 = []		#feature 'a' for digit 4
s1 = []		#feature 'b' for digit 1
s4 = []		#feature 'b' for digit 4
in1=0		#index to point at 'i1' and 's1'
in4=0		#index to point at 'i4' and 's4'

#define our 1st figure for representing the features and plotting them on plane
figure(0)
suptitle("Logistic Regression Digits")
xlabel("X")
ylabel("Y")
for i in range (len(data)) :
	#pick an image separated by new line
    data[i] = data[i].strip("\n")
    #pick individual intensities of image separated by space
    temp = data[i].split(" ")
    #keep only digits '1' and '4' neglect all others
    if (temp[0]=="1") :
        s1.append(0)	#make 'b' feature 0 and add absolute difference of intensities
        i1.append(0)	#make 'a' feature 0 and add simply intensity
        max1 = float(temp[1])	#for finding max intensity
        min1 = float(temp[1])	#for finding min intensity
        for j in range (1,len(temp)) :	#iterate over image matrix and calculate feature
            if(temp[j] != -1) :
                max1 = max(max1,float(temp[j]))
                min1 = min(min1,float(temp[j]))
                i1[in1] = i1[in1] + abs(float(temp[j]))
                s1[in1] = s1[in1] + abs(float(temp[j])-float(temp[257-j]))
        i1[in1] = i1[in1]/256			#find average by dividing by 256
        s1[in1] = (s1[in1]-min1)/(512*(max1-min1))	
        in1 = in1+1
    if (temp[0]=="4") :
        s4.append(0)	#make 'b' feature 0 and add absolute difference of intensities
        i4.append(0)	#make 'a' feature 0 and add simply intensity
        max4 = float(temp[1])	#for finding max intensity
        min4 = float(temp[1])	#for finding min intensity
        for j in range (1,len(temp)) :	#iterate over image matrix and calculate feature
            if(temp[j]!=-1):
                max4 = max(max4,float(temp[j]))
                min4 = min(min4,float(temp[j]))
                i4[in4] = i4[in4] + abs(float(temp[j]))
                s4[in4] = s4[in4] + abs(float(temp[j])-float(temp[257-j]))
        i4[in4] = i4[in4]/256		#find average by dividing by 256
        s4[in4] = (s4[in4]-min4)/(512*(max4-min4))
        in4 = in4+1

#Now our aim is to apply Gradient descent to converge our cost to global minima
x = []					#vector to keep all the features in same vector
w = np.array([0,0,0])	#vector for drawing linear curve for each iteration
w = np.asmatrix(w)		#make it a matrix
alpha = 2				#learning rate
y1 = 1
y4 = -1
epsilon = 1e-9			#to compare with our error

#append features of '1' as [1,'1'.'a','1'.'b']
for i in range(0,len(i1)) :
    temp = [1,i1[i],s1[i]]
    x.append(temp)
#append features of '4' as [1,'4'.'a','4'.'b']
for i in range(0,len(i4)) :
    temp = [1,i4[i],s4[i]]
    x.append(temp)

#convert list 'x' to a numpy matrix
x = np.array(x)
x = np.asmatrix(x)
#set limits on graph
xlim(0.825,0.975)
ylim(-0.01,0.3)
#plot the feature points of both digits
draw_point(i1,s1,1)
draw_point(i4,s4,-1)
error = 0		#set current error as '0'
prev_error = 1	#error in previous iteration
i=0				#iteration counter

#start Gradient descent loop
while((abs(error-prev_error))>epsilon) :
    figure(0)
    prev_error = error
    e=0
    error = 0.0
    for j in range(0,len(i1)) :
        temp = np.asmatrix(x[j])
        sgm1 = sigmoid(-y1,w.transpose(),temp)
        sgm2 = sigmoid(y1,w.transpose(),temp)
        e = e + y1*sgm1*temp
        error = error + math.log(sgm2)
    for j in range(len(i1),len(x)) :
        temp = np.asmatrix(x[j])
        sgm1 = sigmoid(-y4,w.transpose(),temp)
        sgm2 = sigmoid(y4,w.transpose(),temp)
        e = e + y4*sgm1*temp
        error = error + math.log(sgm2)
    error = -error/len(x)
    e = -e/len(x)
    w = w - alpha*e
    if(i!=0) :
        gca().lines[-1].remove()
    show_curve(w)
    print(error)
    figure(1)
    xlabel("Iteration")
    ylabel("Error")
    suptitle("Error plot")
    xlim(0,10000)
    ylim(0,1)
    scatter(i,error,color="black",marker=".",s=1)
    i = i+1
    pause(0.00000000000000000000000001)

show()
