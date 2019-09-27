# Decision-Tree-Generation
Cancer Diagnosis by generating decision tree in PySpark without using any in-built functions<br>

<b>Environment Set Up</b> <br>
It requires Anaconda Jupyter notebook to be installed<br>
Java version above 6 <br>
Apache Spark 2.4.2  <br>


<b>i.Choice of Parameters and attribute selection metric </b><br>
The dataset wdbc.data has been used it consists of 32 attributes ,in the model trained we have used 31 attributes except the ID number. 
Impurity measure = Info Gain(Entropy) <br>
<b>ii.Any assumptions made </b><br>
The attribute Diagnosis (M = malignant, B = benign) is assumed to have the following numerical values for prediction purpose <br>
Malignant(M) = 1.0 <br>
Benign(B) = 0.0  <br>
<b>iii. Validation and Train/Test Strategy used </b><br>
To obtain the classification error in test data, map function is used on the test data and confusion matrix is computed  comparing the actual_label and predicted_label.This is used to find the accuracy by  using the formula <br>
accuracy  =  (TP+TN)/(TP+TN+FP+FN) <br>
Code Snippet: <br>
accuracy=positiveOutput/float(totalOutput) <br>
output: accuracy : 92.12 <br>
