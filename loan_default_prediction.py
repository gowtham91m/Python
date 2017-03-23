import pandas as pd
import os
from View import view
from pandastable import TableModel
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime as dt
#sample = TableModel.getSampleData()

wd = 'C:\\Users\\gmallik\\Downloads\\kaggle\\hackerrank\\loan'
os.chdir(wd)
train_df_org = pd.read_csv('train_indessa.csv')
test_df_org = pd.read_csv('test_indessa.csv')
#train_df = train_df1.sample(10000)

train_df = train_df_org
test_df = test_df_org
#************************************************************************
#feature engineering
'''
# undersample positive outcome to make target variable distribution even

# number of loan default incidents
number_records_default = len(train_df[train_df.loan_status ==1])
default_indices = np.array(train_df[train_df.loan_status == 1].index)

#picking the indices of non default (normal) indices
normal_indices = train_df[train_df.loan_status ==0].index

# out of indices picked, randomly select 'x' number of records
random_normal_indices = np.random.choice(normal_indices,number_records_default,replace=False)
random_normal_indices = np.array(random_normal_indices)

# appending two indices
under_sample_indices = np.concatenate([default_indices,random_normal_indices])

#undersmaple dataset
train_df = train_df.iloc[under_sample_indices,:]
'''
def create_cummy(df):
    for i in df.select_dtypes(include=['object']).columns:
        dummy = pd.get_dummies(df[i],prefix = i)
        df = pd.concat([df,dummy],axis=1)
        df.drop([i],inplace=True,axis=1)
    return df

def feature_eng(df):    
    # delete columns
    df= df.drop(['desc','mths_since_last_delinq','mths_since_last_record',\
                 'mths_since_last_major_derog','verification_status_joint','emp_title',\
                    'title','application_type','pymnt_plan'], axis = 1)
    df = df.drop([ 'zip_code','addr_state','last_week_pay','batch_enrolled'],axis=1)
    #correlation
    df= df.drop(['funded_amnt','funded_amnt_inv'],axis=1)

    # low variance 
    df = df.drop(['total_rec_late_fee','recoveries','collection_recovery_fee',\
                  'collections_12_mths_ex_med','acc_now_delinq'],axis=1)

    #Log transformation for skewed features
    log_columns = ['annual_inc','dti','open_acc','revol_bal','revol_util','total_acc',\
                  'total_rec_int','tot_cur_bal','total_rev_hi_lim']
    for i in log_columns:
        df['log_'+i]=np.log(df[i]+1)
    df = df.drop(log_columns, axis = 1)
    
    # binary encoding for highly skewed numerical values
    bin_col = ['pub_rec','delinq_2yrs','inq_last_6mths','tot_coll_amt']
    for i in bin_col:
        df['if_'+i] = df[i].apply(lambda x: 0 if x==0 else 1)
    df = df.drop(bin_col,axis=1)

    df.fillna(df.ix[:,df.isnull().any()].mean(),inplace=True)
    
    df['emp_length'] = df['emp_length'].map( {'< 1 year': 0,
                                              '1 year' : 1,
                                              '2 years' : 2,
                                              '3 years' : 3,
                                              '4 years' : 4,
                                              '5 years' : 5,
                                              '6 years' : 6,
                                              '7 years' : 7,
                                              '8 years' : 8,
                                              '9 years' : 9,
                                              '10+ years' : 10,
                                              'n/a' : -1}).astype(int)
    df=create_cummy(df)
    
    drop_dummy = ['home_ownership_ANY','home_ownership_NONE','home_ownership_OTHER']
    for i in drop_dummy:
        try:
            df = df.drop([i],axis=1)
        except:
            pass
    df = df.rename(columns={'term_36 months': 'term_36_months', 'term_60 months': 'term_60_months'})
    return df

# low frequency count categories
train_df = train_df[train_df.home_ownership != 'ANY']
train_df = train_df[train_df.home_ownership != 'NONE']
train_df = train_df[train_df.home_ownership != 'OTHER']
train_df.dropna(thresh=23,inplace=True)

train_df = feature_eng(train_df)
test_df = feature_eng(test_df)
    
'''
#dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
#dataset['Title'] = dataset['Title'].fillna(0)
#dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    
# collinerity
#view(train_df.ix[:,1:].corr())

##colormap = plt.cm.viridis
##plt.figure(figsize=(12,12))
##plt.title('Pearson Correlation of Features', y=1.05, size=15)
##sns.heatmap(train_df.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
##plt.show()


# count of unique values
#print(train_df[list(train_df.dtypes.pipe(lambda x:x[x=='object']).index)].apply(lambda x: x.unique().shape[0]))
#print(train_df.select_dtypes(include=['object']).apply(lambda x: x.nunique()))
#print(train_df.loc[:,df.dtypes == object ].apply(pd.Series.nunique))
print(train_df.select_dtypes(include=['object']).apply(pd.Series.nunique))

# frequency table
print(train_df['term'].value_counts())
print(train_df['grade'].value_counts())
print(train_df['sub_grade'].value_counts())
print(train_df['emp_length'].value_counts())
print(train_df['home_ownership'].value_counts())
print(train_df['verification_status'].value_counts())
print(train_df['purpose'].value_counts())
print(train_df['initial_list_status'].value_counts())
print(train_df['loan_status'].value_counts())


# missing values
#print(train_df.apply(lambda x: sum(x.isnull().values), axis=0))
#print(train_df.isnull().sum())
#print(train_df.apply(lambda x: (sum(x.isnull().values))/x.shape[0]*100, axis=0))
#print(train_df.isnull().values.sum())
#print(len(train_df))
#print(train_df.count())
#sum(pd.isnull(df1['col1']))
#print(len(train_df)-train_df.count())

print(train_df.ix[:,train_df.isnull().any()].isnull().sum())
print((len(train_df)-train_df.ix[:,train_df.isnull().any()].count())/len(train_df)*100)
print(train_df.ix[:,train_df.isnull().any()].dtypes)


##print(sum(train_df.apply(lambda x: sum(x.isnull().values),axis=1)>0)) # 42221
##print(sum(train_df.apply(lambda x: sum(x.isnull().values),axis=1)>20))#0
##print(sum(train_df.apply(lambda x: sum(x.isnull().values),axis=1)>10))#16
##print(sum(train_df.apply(lambda x: sum(x.isnull().values),axis=1)>15))#0
##print(sum(train_df.apply(lambda x: sum(x.isnull().values),axis=1)>12))#0
##print(sum(train_df.apply(lambda x: sum(x.isnull().values),axis=1)>11))#3
#df = train_df[train_df.isnull().sum(axis=1)==12]

# drop missing values
#df_no_missing = train_df.dropna() # any nan values

# drop rows where all the cells in that row is NA
#df_cleaned = df.dropna(how = 'all')

#Drop column if they only contain missing values
#df.dropna(axis=1, how='all')

#Drop rows that contain less than 20 observations
#df=train_df.dropna(thresh=23)
#print(train_df.shape[0]-df.shape[0])
#train_df.dropna(thresh=23,inplace=True)


#Fill in missing data with zeros
#df.fillna(0)

# visualization of missing values
#msno.matrix(train_df)
#msno.bar(train_df)


# imputation
# numerical columns
#train_df.fillna(train_df.ix[:,train_df.isnull().any()].mean(),inplace=True)
#train_df_interpolate = train_df.interpolate()

# categorical
#train_df = train_df.apply(lambda x:x.fillna(x.value_counts().index[0]))
##mode(train_df['col'])
##mode(data['Gender']).mode[0]
##data['Gender'].fillna(mode(data['Gender']).mode[0], inplace=True)

view(pd.DataFrame(train_df.columns))

train_df.total_rev_hi_lim.hist()
np.log(train_df.total_rev_hi_lim+1).hist()
plt.show()

print(train_df.total_rev_hi_lim.value_counts())
view(train_df)
print(train_df.shape)
'''

    
#**************************************************************************#
# machine learning

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA as pca
from sklearn.decomposition import FactorAnalysis as fact
from sklearn import cluster as cls
from sklearn.feature_extraction.image import grid_to_graph
from sklearn import tree
from scipy.cluster import hierarchy as hier
from sklearn.externals.six import StringIO
#import xgboost as xgb

from sklearn.naive_bayes import GaussianNB

# train test split
x = train_df.ix[:,train_df.columns!='loan_status']
y = train_df.ix[:,train_df.columns=='loan_status']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state = 0)

lr = LogisticRegression().fit(x_train.ix[:,1:],y_train.values.ravel())
rf = RandomForestClassifier().fit(x_train.ix[:,1:],y_train.values.ravel())
ab = AdaBoostClassifier().fit(x_train.ix[:,1:],y_train.values.ravel())
gb = GradientBoostingClassifier().fit(x_train.ix[:,1:],y_train.values.ravel())
et = ExtraTreesClassifier().fit(x_train.ix[:,1:],y_train.values.ravel())
svm = SVC().fit(x_train.ix[:,1:],y_train.values.ravel())
gnb = GaussianNB().fit(x_train.ix[:,1:],y_train.values.ravel())

print(lr.score(x_test.ix[:,1:],y_test))
print(rf.score(x_test.ix[:,1:],y_test))
print(ab.score(x_test.ix[:,1:],y_test))
print(gb.score(x_test.ix[:,1:],y_test))
print(et.score(x_test.ix[:,1:],y_test))
print(svm.score(x_test.ix[:,1:],y_test))
print(gnb.score(x_test.ix[:,1:],y_test))


#pred_bin=lr_fit.predict(test_df.ix[:,1:])
pred_clf=gb.predict_proba(test_df.ix[:,1:])

pred=pd.DataFrame(pred_clf)[0]
pred=pd.DataFrame(pred)
export = test_df[['member_id']]
export['loan_status']=pred
export.to_csv('export.csv',index=False)
