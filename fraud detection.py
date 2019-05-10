#package importing
import pandas as pd 
import os
os.getcwd()
#reading dataset
df=pd.read_csv('PS_20174392719_1491204439457_log.csv')
#viewing the first 5 rows of the dataset
print (df.head())

#dropping the redundant features
df=df.drop(['nameOrig','nameDest','isFlaggedFraud'],axis=1)

#sorting the fraudulent  and non-fraudulant data into a dataframe
df_fraud=df[df['isFraud']==1]
df_nofraud=df[df['isFraud']==0]
df_nofraud=df_nofraud.head(12000)

#joining both dataset together
df= pd.concat([df_fraud,df_nofraud],axis=0)

#packages importing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#converting the type column to categorical
df['type']=df['type'].astype('category')

#integer encoding the type column
type_encoder=LabelEncoder()
df['type']=type_encoder.fit_transform(df.type)
type_one_hot=OneHotEncoder()
type_one_hot_encode=type_one_hot.fit_transform(df.type.values.reshape(-1,1)).toarray()
#adding the one hot encoded variables to the dataset
ohe_variable=pd.DataFrame(type_one_hot_encode,columns=["type"+str(int(i))for i in range (type_one_hot_encode.shape[1])])
df=pd.concat([df,ohe_variable],axis=1)
#dropping the original type variable
df=df.drop('type',axis=1)
#viewing the new dataframe after one-hot-encoding
print (df.head())
#checking every column for missing values
df.isnull().any()
df=df.fillna(0)
df.to_csv('fraud-detection.csv')
