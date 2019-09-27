import sys 
import math  
from pyspark import SparkContext, SparkConf  
bin=[] 
takenAttributes=[] 
predicts=[] 
##Parsing the input file converting the categorical class to numeric value 
def dataparse(file): 
 filedata=file.split(",") 
 for i in range(len(takenAttributes)): 
  filedata[takenAttributes[i]+2]="discard" 
 data=[] 
 if filedata[1]== 'M': 
  filedata[1]=1.0 
 else: 
  filedata[1]=0.0 
 for x in range(2,len(filedata)): 
  if(filedata[x]!="discard"): 
   value = str(x-2)+'*'+str(filedata[x])+"*"+str(filedata[1]) 
   data.append((value,1)) 
   
 return data 
##Getting data for overall entropy calculation 
def getClassPosNegCount(line): 
 filedata=line.split(",") 
 if(len(filedata)>=2): 
  return (filedata[1],1) 
 else: 
  return "" 
 ##Generating bin value for splitting the data(using 2 bins) 
def createBinValue(file): 
  
# bin=[]  
 for i in range(2,32): 
  min_val= file.map(lambda data: float(data.split(",")[i])).min() 
  max_val= file.map(lambda data: float(data.split(",")[i])).max() 
  bin_val=(float(min_val)+float(max_val))/2 
  bin.append(float(bin_val)) 
 #return bin 
##Generating data for entropy calculation for each attribute 
def calcDataForEntropy(key,val): 
    data1=[] 
 keySplits=key.split("*") 
  return (keySplits[0],keySplits[1]+"*"+keySplits[2]+"*"+str(val)) 
  
  #Reducing values for each attribute 
def reduceEachAttributes(val1,val2): 
 return val1+":"+val2 
##Information Gain calculation 
def calcInfoGain(attribute,value,overallEntropy): 
 inputs=value.split(":") 
 split1=[] 
 split2=[] 
 for i in range(len(inputs)): 
  data=inputs[i].split("*") 
  if(float(data[0])<=bin[int(attribute)]): 
   split1.append(inputs[i]) 
  else: 
   split2.append(inputs[i]) 
 posSplit1=0 
 negSplit1=0 
 posSplit2=0 
 negSplit2=0 
 ##Computing the positive and negative split for each attribute 
 for i in range(len(split1)): 
  data=split1[i].split("*") 
  if(data[1]=='1.0'): 
   posSplit1=posSplit1+1 
  else: 
   negSplit1=negSplit1+1 
  
 for i in range(len(split2)): 
  data=split2[i].split("*") 
  if(data[1]=='1.0'): 
   posSplit2=posSplit2+1 
  else: 
   negSplit2=negSplit2+1 
    
 totalSplit1count=posSplit1+negSplit1 
 totalSplit2count=posSplit2+negSplit2 
 split1Entropy=0 
 split2Entropy=0 
##Entropy Calculation 
 if(posSplit1>0 and negSplit1> 0): 
  log2pos=math.log10(float(posSplit1)/float(totalSplit1count))/math.log10(2.0) 
  log2neg=math.log10(float(negSplit1)/float(totalSplit1count))/math.log10(2.0) 
  split1Entropy=- (((float(posSplit1)/float(totalSplit1count))*log2pos)+((float(negSplit1)/float(totalSplit1count))*log2neg)) 
 else: 
  split1Entropy=0 
  
 if(posSplit2>0 and negSplit2> 0): 
  log2pos=math.log10(float(posSplit2)/float(totalSplit2count))/math.log10(2.0) 
  log2neg=math.log10(float(negSplit2)/float(totalSplit2count))/math.log10(2.0) 
  split2Entropy=- (((float(posSplit2)/float(totalSplit2count))*log2pos)+((float(negSplit2)/float(totalSplit2count))*log2neg)) 
 else: 
  split2Entropy=0 
 totalEntropy=0 
 infoGain=0 
 totalEntropy=((totalSplit1count/(float(totalSplit1count+totalSplit2count)))*split1Entropy) +((totalSplit2count/(float(totalSplit1count+totalSplit2count)))*split2Entropy) 
 if(totalEntropy!=0): 
  infoGain=overallEntropy-totalEntropy 
 else: 
  infoGain=1.0 
 return [infoGain,attribute,posSplit1,negSplit1,posSplit2,negSplit2] 
  
def positiveDataPartition(line,attribute): 
 data=line.split(",") 
 if((float(data[int(attribute)+2]))<=bin[int(attribute)]): 
  return line 
 else: 
  return "" 
def negativeDataPartition(line,attribute): 
 data=line.split(",") 
 if(float(data[int(attribute)+2])>bin[int(attribute)]): 
  return line 
 else: 
  return "" 
##Displaying  the generated decision tree 
def displayOutput():  
  print(" if( "+str(takenAttributes[0])+ " <= "+str(bin[takenAttributes[0]])+")") 
  print("  if("+str(takenAttributes[1])+" <= "+str(bin[takenAttributes[1]])+")") 
  print("   if("+str(takenAttributes[2])+" <= "+str(bin[takenAttributes[2]])+")") 
  print("       predict="+str(predicts[0])) 
  print("   elif("+str(takenAttributes[2])+" > "+str(bin[takenAttributes[2]])+")") 
  print("    predict="+str(predicts[1])) 
  print("  elif("+str(takenAttributes[1])+" > "+str(bin[takenAttributes[1]])+")") 
  print("   if("+str(takenAttributes[3])+" <= "+str(bin[takenAttributes[3]])+")") 
  print("    predict="+str(predicts[2])) 
  print("   elif("+str(takenAttributes[3])+" > "+str(bin[takenAttributes[3]])+")") 
  print("    predict="+str(predicts[3])) 
  print(" elif("+str(takenAttributes[0])+" > "+str(bin[takenAttributes[0]])+")") 
  print("  if("+str(takenAttributes[4])+" <= "+str(bin[takenAttributes[4]])+")") 
  print("   if("+str(takenAttributes[5])+" <= "+str(bin[takenAttributes[5]])+")") 
  print("    predict="+str(predicts[4])) 
  print("   elif("+str(takenAttributes[5])+" > "+str(bin[takenAttributes[5]])+")") 
  print("    predict="+str(predicts[5])) 
  print("  elif("+str(takenAttributes[4])+" > "+str(bin[takenAttributes[4]])+")") 
  print("   if("+str(takenAttributes[6])+" <= "+str(bin[takenAttributes[6]])+")") 
  print("    predict="+str(predicts[6])) 
  print("   elif("+str(takenAttributes[6])+" > "+str(bin[takenAttributes[6]])+")") 
  print("    predict="+str(predicts[7])) 
  print("\n\n")  
##Computing confusion matrix 
def generateconfusionMatrix(line): 
 data=line.split(",") 
 actualClassValue=0 
 if(data[1]=='M'): 
  actualClassValue=1 
 else: 
  actualClassValue=0 
 if(len(takenAttributes)==7 and len(predicts)==8): 
  if(float(data[takenAttributes[0]+2])<=bin[takenAttributes[0]]): 
   if(float(data[takenAttributes[1]+2])<=bin[takenAttributes[1]]): 
    if(float(data[takenAttributes[2]+2])<=bin[takenAttributes[2]]): 
     if(actualClassValue==0 and predicts[0]==0): 
      return ("TP",1) 
     elif(actualClassValue==1 and predicts[0]==1): 
      return ("TN",1) 
     elif(actualClassValue==0 and predicts[0]==1): 
      return("FP",1) 
     elif(actualClassValue==1 and predicts[0]==0): 
      return("FN",1) 
    elif(float(data[takenAttributes[2]+2])>bin[takenAttributes[2]]): 
     if(actualClassValue==0 and predicts[1]==0): 
      return ("TP",1) 
     elif(actualClassValue==1 and predicts[1]==1): 
      return ("TN",1) 
     elif(actualClassValue==0 and predicts[1]==1): 
      return("FP",1) 
     elif(actualClassValue==1 and predicts[1]==0): 
      return("FN",1) 
   elif(float(data[takenAttributes[1]+2])>bin[takenAttributes[1]]): 
    if(float(data[takenAttributes[3]+2])<=bin[takenAttributes[3]]): 
     if(actualClassValue==0 and predicts[2]==0): 
      return ("TP",1) 
     elif(actualClassValue==1 and predicts[2]==1): 
      return ("TN",1) 
     elif(actualClassValue==0 and predicts[2]==1): 
      return("FP",1) 
     elif(actualClassValue==1 and predicts[2]==0): 
      return("FN",1) 
    elif(float(data[takenAttributes[3]+2])>bin[takenAttributes[3]]): 
     if(actualClassValue==0 and predicts[3]==0): 
      return ("TP",1) 
     elif(actualClassValue==1 and predicts[3]==1): 
      return ("TN",1) 
     elif(actualClassValue==0 and predicts[3]==1): 
      return("FP",1) 
     elif(actualClassValue==1 and predicts[3]==0): 
      return("FN",1) 
  elif(float(data[takenAttributes[0]+2])>bin[takenAttributes[0]]): 
   if(float(data[takenAttributes[4]+2])<=bin[takenAttributes[4]]): 
    if(float(data[takenAttributes[5]+2])<=bin[takenAttributes[5]]): 
     if(actualClassValue==0 and predicts[4]==0): 
      return ("TP",1) 
     elif(actualClassValue==1 and predicts[4]==1): 
      return ("TN",1) 
     elif(actualClassValue==0 and predicts[4]==1): 
      return("FP",1) 
     elif(actualClassValue==1 and predicts[4]==0): 
      return("FN",1) 
    elif(float(data[takenAttributes[5]+2])>bin[takenAttributes[5]]): 
     if(actualClassValue==0 and predicts[5]==0): 
      return ("TP",1) 
     elif(actualClassValue==1 and predicts[5]==1): 
      return ("TN",1) 
     elif(actualClassValue==0 and predicts[5]==1): 
      return("FP",1) 
     elif(actualClassValue==1 and predicts[5]==0): 
      return("FN",1) 
   elif(float(data[takenAttributes[4]+2])>bin[takenAttributes[4]]): 
    if(float(data[takenAttributes[6]+2])<=bin[takenAttributes[6]]): 
     if(actualClassValue==0 and predicts[6]==0): 
      return ("TP",1) 
     elif(actualClassValue==1 and predicts[6]==1): 
      return ("TN",1) 
     elif(actualClassValue==0 and predicts[6]==1): 
      return("FP",1) 
     elif(actualClassValue==1 and predicts[6]==0): 
      return("FN",1) 
    elif(float(data[takenAttributes[6]+2])>bin[takenAttributes[6]]): 
     if(actualClassValue==0 and predicts[7]==0): 
      return ("TP",1) 
     elif(actualClassValue==1 and predicts[7]==1): 
      return ("TN",1) 
     elif(actualClassValue==0 and predicts[7]==1): 
      return("FP",1) 
     elif(actualClassValue==1 and predicts[7]==0): 
      return("FN",1)  
##Generating Decision tree       
def recursiveDecisionTreeCreation(trainingData1,counter): 
 #overall entropy Calc 
 overallEntropyArrayRDD=trainingData1.map(getClassPosNegCount).filter(lambda x: x is not None).filter(lambda x: x != "") 
 if(overallEntropyArrayRDD.count()>1): 
  overallEntropyArrayRDD=overallEntropyArrayRDD.reduceByKey(lambda v1,v2:v1+v2) 
 overallEntropyArray=overallEntropyArrayRDD.collect() 
 posCount=0 
 negCount=0 
 overallEntropy=0 
 for i in range(len(overallEntropyArray)): 
  if(i==0): 
   posCount=overallEntropyArray[i][1] 
  else: 
   negCount=overallEntropyArray[i][1] 
 totalCount=posCount+negCount 
 log2pos=math.log10(float(posCount)/float(totalCount))/math.log10(2.0) 
 if(negCount!=0): 
  log2neg=math.log10(float(negCount)/float(totalCount))/math.log10(2.0) 
  overallEntropy=- (((float(posCount)/float(totalCount))*log2pos)+((float(negCount)/float(totalCount))*log2neg)) 
 else: 
  overallEntropy=1.0 
 #if(len(overallEntropyArray)==2): 
 # totalCount=overallEntropyArray[0][1]+overallEntropyArray[1][1] 
 # log2pos=math.log10(float(overallEntropyArray[0][1])/float(totalCount))/math.log10(2.0) 
 # log2neg=math.log10(float(overallEntropyArray[1][1])/float(totalCount))/math.log10(2.0) 
 # overallEntropy=- (((float(overallEntropyArray[0][1])/float(totalCount))*log2pos)+((float(overallEntropyArray[1][1])/float(t otalCount))*log2neg)) 
 #else: 
   
  
 #trainingData1.g 
 trainingdata = trainingData1.flatMap(dataparse).reduceByKey(lambda v1,v2:v1 +v2) 
  
 # abc.saveAsTextFile("/user/chandrmk/abc1") 
 # print(testdata.collect()) 
 inputForEntropy=trainingdata.map(lambda x: calcDataForEntropy(x[0],x[1])).reduceByKey(lambda v1,v2: reduceEachAttributes(v1,v2))  
  
 infoGain=inputForEntropy.map(lambda x:calcInfoGain(x[0],x[1],overallEntropy)).max() 
 takenAttributes.append(int(infoGain[1])) 
 leftPatitionDataRDD=trainingData1.map(lambda x:positiveDataPartition(x,infoGain[1])).filter(lambda x: x is not None).filter(lambda x: x != "") 
 rightPatitionDataRDD=trainingData1.map(lambda x:negativeDataPartition(x,infoGain[1])).filter(lambda x: x is not None).filter(lambda x: x != "") 
 print("maxInfoGain Attribute: "+str(infoGain[1])) 
 if(counter!=3): 
  recursiveDecisionTreeCreation(leftPatitionDataRDD,counter+1) 
  recursiveDecisionTreeCreation(rightPatitionDataRDD,counter+1) 
 predict=0 
 if(counter==3): 
  if(infoGain[2]>infoGain[3]): 
   predict=1 
  else: 
   predict=0 
  predicts.append(predict) 
  if(infoGain[4]>infoGain[5]): 
   predict=1 
  else: 
   predict=0 
  predicts.append(predict) 
 #val=infoGain.collect() 
 #for i in range(len(val)): 
 # print(val[i])   
conf = SparkConf().setAppName("Decision Tree") 
sc = SparkContext(conf=conf) 
sc.setLogLevel("Error") 
       #input 
file = sc.textFile("wdbc.data") 
##Splitting of data as train and test 
(trainingData1, testData1) = file.randomSplit([0.7, 0.3]) 
createBinValue(trainingData1) 
#for i in range(len(bin)): 
# print(bin[i])  
recursiveDecisionTreeCreation(trainingData1, 1)  
matrix=testData1.map(generateconfusionMatrix).reduceByKey(lambda v1,v2:v1+v2).collect() 
displayOutput() 
positiveOutput=0 
totalOutput=0 
for i in range(len(matrix)): 
 if( matrix[i][0]=='TP' or matrix[i][0]=='TN'): 
  positiveOutput= positiveOutput+int(matrix[i][1]) 
 totalOutput= totalOutput + int(matrix[i][1]) 
accuracy=positiveOutput/float(totalOutput)  
print(matrix) 
print("Accuracy : "+str(accuracy)) 
