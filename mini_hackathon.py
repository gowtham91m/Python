import pandas as pd
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier

df=pd.read_csv('train_63qYitG.csv')
test = pd.read_csv('test_XaoFywY.csv')
#print(df.shape)
#print(df.dtypes)

# total number of missing values
print('total number of missing values')
print(sum(df.isnull().values.ravel()))

# counting rows that have missing value somewhere
print('rows that have missing values somewhere')
print(sum([True for idx,row in df.iterrows() if any(row.isnull())]))

# missing values by column
print('columns with missing values')
print(df.apply(lambda x: sum(x.isnull().values), axis = 0))
print(test.apply(lambda x: sum(x.isnull().values), axis = 0))


#delete trip id column
del df['Trip_ID']
#print(list(df.columns))


# convert categorical to numeric
#gender
Gender = pd.Series(np.where(df.Gender == 'Male',1,0),name='Gender')
df.Gender = Gender

tGender = pd.Series(np.where(test.Gender == 'Male',1,0),name='Gender')
test.Gender=tGender

#type of cab
Type_of_cab = pd.get_dummies(df.Type_of_Cab, prefix='Type_of_cab')
tType_of_cab = pd.get_dummies(test.Type_of_Cab, prefix='Type_of_cab')

Confidence_Life_Style_Index =pd.get_dummies(df.Confidence_Life_Style_Index, prefix = 'Confidence_Life_Style_Index')
tConfidence_Life_Style_Index =pd.get_dummies(test.Confidence_Life_Style_Index, prefix = 'Confidence_Life_Style_Index')

Destination_Type = pd.get_dummies(df.Destination_Type, prefix='Destination_Type')
tDestination_Type = pd.get_dummies(test.Destination_Type, prefix='Destination_Type')

df = pd.concat([df,Type_of_cab,Confidence_Life_Style_Index,Destination_Type],axis=1)
test = pd.concat([test,tType_of_cab,tConfidence_Life_Style_Index,tDestination_Type],axis=1)

tcolumns = test.columns.tolist()
tcolumns = [c for c in tcolumns if c not in ['Type_of_Cab','Confidence_Life_Style_Index','Destination_Type']]
test = test[tcolumns]


# simple missing vlue imputation
df.fillna(method='ffill',inplace=True)
test.fillna(method='ffill',inplace=True)

test['Var1']=test.Var1.fillna(test.Var1.mean())

# classifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.cross_validation import train_test_split

#y = df.Surge_Pricing_Type
#x= df.ix[:,:12]

columns = df.columns.tolist()
columns = [c for c in columns if c not in ['Surge_Pricing_Type','Type_of_Cab','Confidence_Life_Style_Index','Destination_Type']]

target = 'Surge_Pricing_Type'
#print(df[columns].columns)



train = df.sample(frac=0.75, random_state = 1)
val = df.loc[~df.index.isin(train.index)]

clf = DecisionTreeClassifier()
clf = clf.fit(df[columns],df[target])
#clf.score(val[columns],val[target])
print(clf.score(train[columns],train[target]))

val_predict = clf.predict(val[columns])
print(pd.DataFrame(val_predict))

val.to_csv('val.csv')

predict = clf.predict(test.ix[:,1:32])


predict= pd.DataFrame(predict)
predict = pd.concat([test.Trip_ID,predict],axis=1)


predict.to_csv('Sample_Submission.csv',index=False)


