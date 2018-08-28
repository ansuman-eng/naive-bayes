import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB

df=pd.read_csv('SMSSpamCollection',sep='\t',names=['status','message'])		#read the data
df.loc[df["status"]=="ham","status"]=1								#numerical classfication
df.loc[df["status"]=="spam","status"]=0
df_x=df["message"]
df_y=df["status"]

x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.3,random_state=4)

cv=CountVectorizer()
mnb=MultinomialNB()

x_train_cv=cv.fit_transform(x_train)
x_test_cv=cv.transform(x_test)
y_train=y_train.astype('int')

mnb.fit(x_train_cv,y_train)
predictions=mnb.predict(x_test_cv)
count=0
y_test=np.array(y_test)

for i in range(len(predictions)):
	if(predictions[i]==y_test[i]):
		count+=1

print(count)
print(len(predictions))
accuracy=float(count)/len(predictions)
print(accuracy)