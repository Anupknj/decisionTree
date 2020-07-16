import numpy as np
import pandas as pd
eps = np.finfo(float).eps
from numpy import log2 as log
from collections import Counter
import copy
import pprint

# -----Below function are for formatting the dataset recieved.
#  It recieves in datasets in the form of CSV and returns dataframe which has columns for ID, attributes (1 to 60) and associated class

# function formatData returns dataset including class
def formatData(dataset):
    thisdict = {
      
    }  
    # dataset is converted to dictionary with keyword of ID, attributes(1,60) and class                      
    lengthOfData = len(dataset[0]) 
    print(len(dataset[0]))
    serialTemp = list()
    classTemp = list ()
    # ID and class are pushed to dictionary
    for x in range(1,lengthOfData+1):      
        serialTemp.append(dataset[0][x-1])
        classTemp.append(dataset[2][x-1])

    # recursively attributes are parsed into dictionary
    def returnAttributeList(i):
        localList = []
        for x in range (0,lengthOfData):
                localList.append(dataset[1][x][i-1])
        return localList    
    def getAllAttribute():
        returnValue = []
        for number in range(1,61):
            returnValue = returnAttributeList(number)
            thisdict[number]= returnValue
    getAllAttribute()   
    thisdict["ID"] = serialTemp
    thisdict["class"] = classTemp
    # finally dictionary with dataset is returned
    return thisdict

# this function operates same as above.But returns dictionary without class.
# This is used to get required format as said in kaggle
def formatDataFT(datasetFT):
    thisdict = {
      
    }
    lengthOfData = len(datasetFT[0]) 
    print(len(datasetFT[0]))
    serialTemp = list()
    for x in range(1,lengthOfData+1):
        serialTemp.append(datasetFT[0][x-1])
    def returnAttributeList(i):
        localList = []
        for x in range (0,lengthOfData):
                localList.append(datasetFT[1][x][i-1])
        return localList    
    def getAllAttribute():
        returnValue = []
        for number in range(1,61):
            returnValue = returnAttributeList(number)
            thisdict[number]= returnValue
    getAllAttribute()   
    thisdict["ID"] = serialTemp
    return thisdict




# ----Below functions are designed to find gini index and return lowest of all
# function findGini 
def findGini (variable, attribute,den):
    Class = "class"   # it returns "class" keyword
    target_variables = df[Class].unique()   # it returns N, IE, EI
    ginisum = 0
    for target in target_variables:  
        num = len(df[attribute][df[attribute]==variable][df[Class] ==target]) 
        # returns number of combination of attribute and class
        ginisum += (num/den) ** 2  #applies gini index formula
    return 1-ginisum 

def findGiniIndexOfAttribute(df,attribute):
    
    variables = df[attribute].unique() # returns A G T C
    for variable in variables: 
        fraction = 0
        den = len(df[attribute][df[attribute]==variable]) # calculates number of A, G, T, C
        fraction +=den/len(df) * findGini(variable,attribute,den) #it calculates gini index of all attributes
    return fraction

# ----Below code is for miscalculation error

def findmisEr (variable, attribute,den):
    Class = "class"   # it returns "class" keyword
    target_variables = df[Class].unique()   # it returns N, IE, EI
    maxProbability = 0
    for target in target_variables:  

        num = len(df[attribute][df[attribute]==variable][df[Class] ==target]) 
        # returns number of combination of attribute and class
        if ( (num/den)> maxProbability):  # selects one with max prob
            maxProbability = (num/den)  #applies mis erroe formula
        
    return 1-maxProbability 

def findmisErIndexOfAttribute(df,attribute):
    
    variables = df[attribute].unique() # returns A G T C
    for variable in variables: 
        fraction = 0
        den = len(df[attribute][df[attribute]==variable]) # calculates number of A, G, T, C
        fraction +=den/len(df) * findmisEr(variable,attribute,den) #it calculates  probability of all attributes
    return fraction

#----Below functions are meant for entropy and IG calulation

def findEntropy(df):
    Class = "class"   # returns name "class"
    entropy = 0
    values = df[Class].unique() # returns classes as N IE EI
    for value in values:
        fraction = df[Class].value_counts()[value]/len(df[Class]) 
        entropy += -fraction*np.log2(fraction) #using entropy formula
    return entropy
  
def findEntropyAttribute(df,attribute):
  Class = df.keys()[-1]   # returns keyword class
  target_variables = df[Class].unique()  # returns N, IE, EI
  variables = df[attribute].unique()   # returns attributes A G T C
  entropy2 = 0
  for variable in variables:
      entropy = 0
      for target_variable in target_variables:
          num = len(df[attribute][df[attribute]==variable][df[Class] ==target_variable]) # returns number of specific attribute and target class
          den = len(df[attribute][df[attribute]==variable])  #  returns number attribute with specific value
          fraction = num/(den+eps)
          entropy += -fraction*log(fraction+eps) # using entropy formula
      fraction2 = den/len(df)
      entropy2 += -fraction2*entropy
  return abs(entropy2) # returns final value


#----Below functions for finding max or min value for entropy and gini respectively
def findEntropyWinner(df):
    IG = [] # holds information gain values
    for key in df.keys()[:-1]:
        IG.append(findEntropy(df)-findEntropyAttribute(df,key))  # entire entropy - entropy for specific attribute
    return df.keys()[:-1][np.argmax(IG)] # returns max value for building tree
  
def find_gini_winner(df):
    GI = [] # gini index array 
    for key in df.keys()[:-1]:
        GI.append(findGiniIndexOfAttribute(df,key)) # calculates gini indes for each attribute values
    return df.keys()[:-1][np.argmin(GI)] # returns lowest value for building tree

def find_misEr_winner(df):
    misEr = [] # gini index array 
    for key in df.keys()[:-1]:
        misEr.append(findmisErIndexOfAttribute(df,key)) # calculates probability for each attribute values
    return df.keys()[:-1][np.argmin(misEr)] # returns lowest value for building tree


##-------- Below  is the code for chi square pruning

# gets unique keys for creating table ie, A G T C
def getUnique(list):
    values = Counter(list).keys()
    output = []
    for x in values:    
        output.append(x)
    return output

# counts the occurence of combination of AGTC and N IE EI to create 2 dimensional array
def countValues(list1, list2, first, second):
    counter = 0
    for i in range(len(list1)):
        if first == list1[i] and second == list2[i]:
            counter = counter + 1
    return counter

def findExpectedOfNode(node, saveFlag = 0):
    global expectedNumber
    global lenExpected
    global lenActual

    columnList = df[node].values.tolist()
    lenActual = len(columnList)
    classList = df['class'].values.tolist()

    listOne = getUnique(columnList)   # A G T C
    listTwo = getUnique(classList)   # N IE EI

    expectedNumberTemp = []
    for i in listOne:   # recurses through AGTC
        row = []
        for j in listTwo: # recurses through N EI IE
            row.append(countValues(columnList, classList, i, j))
        expectedNumberTemp.append(row)
    if saveFlag == 1:  # when this flag is saved expected number of this is saved in global variable for future comparisons
        expectedNumber = copy.deepcopy(expectedNumberTemp)
        lenExpected = len(columnList)
    return expectedNumberTemp

def makeList(number):
    result = []
    for i in range(number):
        result.append(i)

# this calculates chi square value of the node using the formula (observed - expected) **2 / expected

def chiSquare(node):  
    xSquarevalue = 0
    counterx = 0
    currentValue = findExpectedOfNode(node)
    for i in range (len(expectedNumber)): # recurces through A G T C
        for j in range (len(expectedNumber[0])): # recurses through N EI  IE
            try:
                counterx += 1      
                # uses the chi square formula 
                xSquarevalue += (((currentValue[i][j]/lenActual)-(expectedNumber[i][j]/lenExpected))**2) / (expectedNumber[i][j]/lenExpected)
            except:
                pass
    return xSquarevalue    

# evaluates whether node chisquare value is greater or smaller than confidential value
def chiSquareTestPruning(node,confidentialValue):
    global firstNode
    if firstNode == 1:          # if this is first node of the tree
        findExpectedOfNode(node,1) # its expected number is saved
        firstNode = 0           # first node flag is set to false
        return 1
    else:
        checkValue = chiSquare(node)  # returns check value for the node

        if confidentialValue == dOF["0.05"]: # considering the degree of freedom values are checked
            if checkValue > 12.592:   # nodes are returned if it is smaller than the value else rejected
                return 0
            if checkValue <= 12.592:    
                return 1

        elif confidentialValue == dOF["0.01"]:
            if checkValue > 16.812:
                return 0
            if checkValue <= 16.812:
                return 1

# it checks whether node is appropraite for pruning

def getAppropriateNode(df):
    rootNode = findEntropyWinner(df) # find node with highest eIG
    node = chiSquareTestPruning(rootNode,accuracyLevel)
    if node == 0:   # if node is not appropriate
        getAppropriateNode(df) #checks for next node
    if node == 1:   # if node is appropraite
        node = rootNode
        findExpectedOfNode(node,1) # returns expected number and if flag is set, then expected number is saved in global variable
        return rootNode

# ---Below is the code for tree
def get_subtable(df, node,value):
  return df[df[node] == value].reset_index(drop=True) # helps to recurse the tree

def buildTree(df,tree=None): 
    node = 0
    attValue = list()

    if giniFlag == 1:
        node = find_gini_winner(df)

    if entropyFlag == 1:
        node = findEntropyWinner(df)
    
    if chiSquareFlag == 1:
        node = getAppropriateNode(df)
    
    if miscalculationErrorFlag ==1:
        node = find_misEr_winner(df)


    # gets node from respected method and uses to build the tree
    try:
        attValue = np.unique(df[node])
    except:
        print("An exception occurred")

    # create base tree
    if tree is None:                    
        tree={}
        tree[node] = {}
    
    #We make loop to construct a tree by calling this function recursively. 
    #In this we check if the subset is pure and stops if it is pure. 
    
    for value in attValue:
        subtable = get_subtable(df,node,value)
        clValue,counts = np.unique(subtable['class'],return_counts=True)                        
        if len(counts)==1:                  #Checking purity of subset
            tree[node][value] = clValue[0]                                                    
        else:        
            tree[node][value] = buildTree(subtable) #Calling the function recursively 
                   
    return tree

    #This function is used to predict for any input variable 
    
    #Recursively we go through the tree that we built earlier
#--- Below is the code for prediction and finding accuracy
def predict(inst,tree):
    for nodes in tree.keys():   
        value = inst[nodes]
        if value not in tree[nodes]:
            prediction = 'N'  # unfollowed sequence is hard coded to N class
            break
        tree = tree[nodes][value]
        prediction = 0
        if type(tree) is dict:
            prediction = predict(inst, tree)
        else:
            prediction = tree
            break   
    return prediction  # returns the pridiction

def findAccuracy(df2):
    passedCases=0
    failedCases=0
    for i in range(0,len(df2)):   # passses the dataframe df2 for predictiona nd calculating accuracy
        prediction = predict(df2.iloc[i],tree)  
        trueClass = df2.iloc[i]['class']
        if prediction == trueClass:
            passedCases+=1
        else:
            failedCases+=1
    accuraccy = passedCases/(passedCases+failedCases) * 100  # returns the accuracy
    return accuraccy

# this function uses dataframe for testing ( format is as specified in kaggle) and returns list of lists having ID and classes
def accuraccyCSV(dfFT):
    for i in range(0,len(dfFT)):
        tempList=[]
        prediction = predict(dfFT.iloc[i],tree)
        tempList.append(dfFT.iloc[i]["ID"])
        tempList.append(prediction)
        listToDF.append(tempList)
    return listToDF  


# this function gets node(1-60) and  attribute(AGTC) and returns max number of classes where attribute is found in that node 
def getHighProbableTarget(node,attribute):
    nN= nIE =nEI =0
    for i in range (1,len(df)):
        if df.iloc[i][node] == attribute :
            tempClass = df.iloc[i]["class"]
            if tempClass == "N":
                nN+=1
            if tempClass == "EI":
                nEI+=1
            if tempClass == "IE":
                nIE+=1

    if nN > nEI:
        if nN > nIE:
            return "N"
        else:
            return "IE"
    else:
        if nEI > nIE:
            return "EI"
        else:
            return "IE"




if __name__ == '__main__':
    # reads data from csv file
    dataset = pd.read_csv (r'training.csv', squeeze=True, header=None).to_dict()
    datasetFT = pd.read_csv (r'testing.csv', squeeze=True, header=None).to_dict()

    expectedNumber = [] 
    firstNode = 1 # flag for keeping track of first node
    
    totalLengthOfTable =0
    # flags for different methods of calculation. 
    entropyFlag = 1
    giniFlag = 0
    miscalculationErrorFlag = 0
    chiSquareFlag = 0
    # degree of freedom is hard coded in to a dictionary
    dOF = {
    "0.01": 16.812,
    "0.05": 12.592   
    }

    # default accuracy level is set to 0.01
    accuracyLevel = "0.01"
    # format dataset to convert it  into dictionary with keywords ID ,class, and attributes (1-60)
    formattedDataset = formatData(dataset)
    formattedDatasetFT = formatDataFT(datasetFT)

    a = list(range(1, 61))
    a.append("class")
    a.append("ID")
    # dictionaries are converted to dataframe
    df = pd.DataFrame(formattedDataset,columns=a)
    dfFT = pd.DataFrame(formattedDatasetFT,columns=a)

    lengthOfSplit = len(df)/2

    # intialize 2 datasets. training set and testing set
    df1 = pd.DataFrame(columns=a)
    df2 = pd.DataFrame(columns=a)

    # split data into 1/2 :1/2
    for i in range(0,len(df)):
   # df1.append(df.iloc[i]) if i<lengthOfSplit else df2.append(df.iloc[i]) 

        if i<lengthOfSplit:
            first = df.iloc[i]
            df1=df1.append(first)

        elif i>=lengthOfSplit:
            second = df.iloc[i] 
            df2=df2.append(second)

    del df1['ID']

    # train using df1 
    tree = buildTree(df1)
    pprint.pprint(tree)

    passedCases=0
    failedCases=0

    # checks the accuracy
    accuracy_score = findAccuracy(df2)

    print("Accuracy is ",accuracy_score)


    listToDF=[]

    # to get csv file for testing accuracy
    finalOutput = accuraccyCSV(dfFT)
    final =pd.DataFrame(finalOutput,columns=['ID','class'])
    final.to_csv('final.csv', index = False)

    


            


        
    






