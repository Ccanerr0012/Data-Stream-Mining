#Caner Canlıer 21702121
import pandas as pd
import numpy as np
from skmultiflow.data import FileStream
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.lazy import KNNClassifier
from skmultiflow.bayes import NaiveBayes
from sklearn.ensemble import VotingClassifier
from skmultiflow.meta import DynamicWeightedMajorityClassifier
from skmultiflow.meta import BatchIncrementalClassifier
import skmultiflow.data 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from combo.models.classifier_stacking import Stacking

#QUESTION 1
"""Dataset Generation"""

#Part a
hyperplane_generator = skmultiflow.data.HyperplaneGenerator(n_features=10, noise_percentage=0.10, n_drift_features=2)
dataset1 = hyperplane_generator.next_sample(20000)
shaped1=np.reshape(dataset1[1],(20000,1))
concated1=np.append(dataset1[0],shaped1,axis=1)
df1 = pd.DataFrame(concated1)
df1.to_csv("C:/Users/ccane/Desktop/ge461/PROJECT 5/Hyperplane Dataset 10_2.csv", index = False,header=False)
print('DataFrame is written successfully to Excel Sheet.')


#Part b
hyperplane_generator = skmultiflow.data.HyperplaneGenerator(n_features=10, noise_percentage=0.30, n_drift_features=2)
dataset2 = hyperplane_generator.next_sample(20000)
shaped2=np.reshape(dataset2[1],(20000,1))
concated2=np.append(dataset2[0],shaped2,axis=1)

df2 = pd.DataFrame(concated2)
df2.to_csv("C:/Users/ccane/Desktop/ge461/PROJECT 5/Hyperplane Dataset 30_2.csv", index=False,header=False)
print('DataFrame is written successfully to Excel Sheet.')

#Part c
hyperplane_generator = skmultiflow.data.HyperplaneGenerator(n_features=10, noise_percentage=0.10, n_drift_features=5)
dataset3 = hyperplane_generator.next_sample(20000)
shaped3=np.reshape(dataset3[1],(20000,1))
concated3=np.append(dataset3[0],shaped3,axis=1)

df3 = pd.DataFrame(concated3)
df3.to_csv("C:/Users/ccane/Desktop/ge461/PROJECT 5/Hyperplane Dataset 10_5.csv", index=False,header=False)
print('DataFrame is written successfully to Excel Sheet.')

#Part d
hyperplane_generator = skmultiflow.data.HyperplaneGenerator(n_features=10, noise_percentage=0.30, n_drift_features=5)
dataset4 = hyperplane_generator.next_sample(20000)
shaped4=np.reshape(dataset4[1],(20000,1))
concated4=np.append(dataset4[0],shaped4,axis=1)

df4 = pd.DataFrame(concated4)
df4.to_csv("C:/Users/ccane/Desktop/ge461/PROJECT 5/Hyperplane Dataset 30_5.csv", index=False,header=False)
print('DataFrame is written successfully to Excel Sheet.')

#QUESTION 2
"""Data Stream Classification"""
data_stream1 =FileStream("C:/Users/ccane/Desktop/ge461/PROJECT 5/Hyperplane Dataset 10_2.csv")
data_stream2 =FileStream("C:/Users/ccane/Desktop/ge461/PROJECT 5/Hyperplane Dataset 30_2.csv")
data_stream3 =FileStream("C:/Users/ccane/Desktop/ge461/PROJECT 5/Hyperplane Dataset 10_5.csv")
data_stream4 =FileStream("C:/Users/ccane/Desktop/ge461/PROJECT 5/Hyperplane Dataset 30_5.csv")
# HoeffdingTree as HT online learner
ht = HoeffdingTreeClassifier()
# Set the evaluator
model = EvaluatePrequential(max_time=1000,show_plot=True,metrics=['accuracy'],data_points_for_classification=True)
#Evaluate
model.evaluate(stream=data_stream1, model=[ht], model_names=['HT'])
model.evaluate(stream=data_stream2, model=[ht], model_names=['HT'])
model.evaluate(stream=data_stream3, model=[ht], model_names=['HT'])
model.evaluate(stream=data_stream4, model=[ht], model_names=['HT'])


#K nearest neighbour as KNN online learner
knn = KNNClassifier()
#Evaluate
model.evaluate(stream=data_stream1, model=[knn], model_names=['KNN'])
model.evaluate(stream=data_stream2, model=[knn], model_names=['KNN'])
model.evaluate(stream=data_stream3, model=[knn], model_names=['KNN'])
model.evaluate(stream=data_stream4, model=[knn], model_names=['KNN'])

#Naïve Bayes as NB online learner
nb= NaiveBayes()
#Evaluate
model.evaluate(stream=data_stream1, model=[nb], model_names=['NB'])
model.evaluate(stream=data_stream2, model=[nb], model_names=['NB'])
model.evaluate(stream=data_stream3, model=[nb], model_names=['NB'])
model.evaluate(stream=data_stream4, model=[nb], model_names=['NB'])


#Compare ALL
model = EvaluatePrequential(max_time=1000,show_plot=True,metrics=['accuracy'])

model.evaluate(stream=data_stream1, model=[ht,knn,nb], model_names=['HT',"KNN","NB"])
model.evaluate(stream=data_stream2, model=[ht,knn,nb], model_names=['HT',"KNN","NB"])
model.evaluate(stream=data_stream3, model=[ht,knn,nb], model_names=['HT',"KNN","NB"])
model.evaluate(stream=data_stream4, model=[ht,knn,nb], model_names=['HT',"KNN","NB"])


#QUESTION 3
"""Data Stream Classification"""
models= [("HT",ht),("KNN",knn),("NB",nb)]
MV = VotingClassifier(estimators=models,voting="hard")
MV = BatchIncrementalClassifier(base_estimator=MV, n_estimators=3)
model.evaluate(stream=data_stream1, model=[MV], model_names=["MV"])

models= [ht,knn,nb]
for types in models:
    WMV= DynamicWeightedMajorityClassifier(base_estimator=types)
    model.evaluate(stream=data_stream1, model=[WMV], model_names=["MV"])

model.evaluate(stream=data_stream1, model=[MV, WMV], model_names=["MV", "WMV"])
model.evaluate(stream=data_stream2, model=[MV, WMV], model_names=["MV", "WMV"])
model.evaluate(stream=data_stream3, model=[MV, WMV], model_names=["MV", "WMV"])
model.evaluate(stream=data_stream4, model=[MV, WMV], model_names=["MV", "WMV"])

#Question 4


#QUESTION 3
"""Data Stream Classification"""
models= [("HT",ht),("KNN",knn),("NB",nb)]
MV = VotingClassifier(estimators=models,voting="hard")
MV = BatchIncrementalClassifier(base_estimator=MV, n_estimators=3)
model.evaluate(stream=data_stream1, model=[MV], model_names=["MV"])

models= [ht,knn,nb]
for types in models:
    WMV= DynamicWeightedMajorityClassifier(base_estimator=types)
    model.evaluate(stream=data_stream1, model=[WMV], model_names=["MV"])

model.evaluate(stream=data_stream1, model=[MV, WMV], model_names=["MV", "WMV"])
model.evaluate(stream=data_stream2, model=[MV, WMV], model_names=["MV", "WMV"])
model.evaluate(stream=data_stream3, model=[MV, WMV], model_names=["MV", "WMV"])
model.evaluate(stream=data_stream4, model=[MV, WMV], model_names=["MV", "WMV"])

"""part-b-) using Interleaved-Test-Then-Train approach"""

df1_features = df1.values[:,:10]
df1_labels = df1.values[:,10]
train_features1, test_features1, train_labels1, test_labels1 = train_test_split(df1_features, df1_labels, test_size=0.3)


df2_features = df2.values[:,:10]
df2_labels = np.array(df2.values[:,10], dtype=int)
train_features2, test_features2, train_labels2, test_labels2 = train_test_split(df2_features, df2_labels, test_size=0.3)

df3_features = df3.values[:,:10]
df3_labels = np.array(df3.values[:,10], dtype=int)
train_features3, test_features3, train_labels3, test_labels3 = train_test_split(df3_features, df3_labels, test_size=0.3)

df4_features = df4.values[:,:10]
df4_labels = np.array(df4.values[:,10], dtype=int)
train_features4, test_features4, train_labels4, test_labels4 = train_test_split(df4_features, df4_labels, test_size=0.3)


ht.fit(train_features1, train_labels1)
predict_ht = ht.predict(test_features1)
cm = confusion_matrix(test_labels1,predict_ht)
accuracy = float(cm.diagonal().sum())/len(test_labels1)
print('ht accuracy is:',accuracy*100,'%')


ht.fit(train_features2, train_labels2)
predict_ht = ht.predict(test_features2)
cm = confusion_matrix(test_labels2,predict_ht)
accuracy = float(cm.diagonal().sum())/len(test_labels2)
print('ht accuracy is:',accuracy*100,'%')

ht.fit(train_features3, train_labels3)
predict_ht = ht.predict(test_features3)
cm = confusion_matrix(test_labels3,predict_ht)
accuracy = float(cm.diagonal().sum())/len(test_labels3)
print('ht accuracy is:',accuracy*100,'%')

ht.fit(train_features4, train_labels4)
predict_ht = ht.predict(test_features4)
cm = confusion_matrix(test_labels4,predict_ht)
accuracy = float(cm.diagonal().sum())/len(test_labels4)
print('ht accuracy is:',accuracy*100,'%')


knn.fit(train_features1, train_labels1)
predict_ht = knn.predict(test_features1)
cm = confusion_matrix(test_labels1,predict_ht)
accuracy = float(cm.diagonal().sum())/len(test_labels1)
print('knn accuracy is:',accuracy*100,'%')


knn.fit(train_features2, train_labels2)
predict_ht = knn.predict(test_features2)
cm = confusion_matrix(test_labels2,predict_ht)
accuracy = float(cm.diagonal().sum())/len(test_labels2)
print('knn accuracy is:',accuracy*100,'%')

knn.fit(train_features3, train_labels3)
predict_ht = knn.predict(test_features3)
cm = confusion_matrix(test_labels3,predict_ht)
accuracy = float(cm.diagonal().sum())/len(test_labels3)
print('knn accuracy is:',accuracy*100,'%')

knn.fit(train_features4, train_labels4)
predict_ht = knn.predict(test_features4)
cm = confusion_matrix(test_labels4,predict_ht)
accuracy = float(cm.diagonal().sum())/len(test_labels4)
print('knn accuracy is:',accuracy*100,'%')

nb.fit(train_features1, train_labels1)
predict_ht = nb.predict(test_features1)
cm = confusion_matrix(test_labels1,predict_ht)
accuracy = float(cm.diagonal().sum())/len(test_labels1)
print('nb accuracy is:',accuracy*100,'%')


nb.fit(train_features2, train_labels2)
predict_ht = nb.predict(test_features2)
cm = confusion_matrix(test_labels2,predict_ht)
accuracy = float(cm.diagonal().sum())/len(test_labels2)
print('nb accuracy is:',accuracy*100,'%')

nb.fit(train_features3, train_labels3)
predict_ht = nb.predict(test_features3)
cm = confusion_matrix(test_labels3,predict_ht)
accuracy = float(cm.diagonal().sum())/len(test_labels3)
print('nb accuracy is:',accuracy*100,'%')

nb.fit(train_features4, train_labels4)
predict_ht = nb.predict(test_features4)
cm = confusion_matrix(test_labels4,predict_ht)
accuracy = float(cm.diagonal().sum())/len(test_labels4)
print('nb accuracy is:',accuracy*100,'%')

batch_size=[1,100,1000]
for batch in batch_size:
    model = EvaluatePrequential(batch_size=batch,show_plot=True, metrics=["accuracy"])

    model.evaluate(stream=data_stream1, model=[ht,knn,nb], model_names=['HT',"KNN","NB"])
    model.evaluate(stream=data_stream2, model=[ht,knn,nb], model_names=['HT',"KNN","NB"])
    model.evaluate(stream=data_stream3, model=[ht,knn,nb], model_names=['HT',"KNN","NB"])
    model.evaluate(stream=data_stream4, model=[ht,knn,nb], model_names=['HT',"KNN","NB"])

# initialize a group of base classifiers
classifiers = [ht,knn,nb]
clf = Stacking(base_estimators=classifiers) 
#Calculate accuracy of clf
clf.fit(train_features1, train_labels1)
predict_ht = clf.predict(test_features1)
cm = confusion_matrix(test_labels1,predict_ht)
accuracy = float(cm.diagonal().sum())/len(test_labels1)
print('clf accuracy is:',accuracy*100,'%')

