import numpy as np
from matplotlib import pyplot as plt
import random as rn
import math
from pylab import *

import csv 

def calculate_entropy(df, features, target):
	

def ID3_util(df, features, target):
	entropy_H = calculate_entropy(df,features,target)


def ID3(df):
	features = list(df)
	ID3_util(df,features,features[-1])
# csv file name 
filename = "Vegetation.csv"
  
import pandas
df = pandas.read_csv(filename)
features = list(df)
print(list(df))
ID3(df)