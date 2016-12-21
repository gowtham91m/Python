# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 15:42:20 2016
@author: Gowtham Mallikarjuna
"""
import os
import pandas as pd
#pd.set_option('display.max_columns', None) #dilplay all teh columns
#pd.set_option('display.max_rows', None) #dilplay all teh columns
#pd.set_option('display.expand_frame_repr', None) #dilplay all teh columns

# get location of script
# This works only on command prompt. If working on any IDE, set current woking directory manually
wd = os.path.dirname(os.path.realpath(__file__))
# change working directory to script location
#wd='C:\\Users\\gmallik\\Downloads\\OSUAA\\merge\\merge'
os.chdir(wd)
#create directory 'export' if not exists
if not os.path.isdir('export'):
    os.mkdir('export')
# set relative path from working directory
export_path = wd +'\\export\\'
data_path = wd+'\\data\\'
recon_path=wd+'\\reconciliation\\'
files=os.listdir(data_path)
recon=os.listdir(recon_path)

# aggridage and store all reconciliation files
recon_data = 0
for k in recon:
    if k[-4:]=='.csv':
        df= pd.read_csv(recon_path+k,parse_dates=['CnBio_Birth_date'])
        df.rename(columns={'First Name':'First_Name','Last Name':'Last_Name'},inplace=True)
        df = df.ix[:,['CnBio_ID','First_Name','Last_Name','CnBio_Birth_date']]
        df['Month']=k[:3]
        if type(recon_data) is type(0):
            recon_data=df
        else:
            recon_data=recon_data.append(df)
recon_data = recon_data[(pd.isnull(recon_data['CnBio_ID'])==False)]
recon_data.drop_duplicates(['First_Name','Last_Name','CnBio_ID','CnBio_Birth_date'],inplace=True)
recon_data['First_Name']=pd.core.strings.str_strip(recon_data['First_Name'])
recon_data['Last_Name']=pd.core.strings.str_strip(recon_data['Last_Name'])
# Merge weekly membership transaction data with reconciliationfiles to get cn_BioID
for i in files:
    if i[-4:]=='.csv':
        df= pd.read_csv(data_path+str(i),parse_dates=['CnBio_Birth_date'],skipinitialspace=True)
        df.rename(columns={'Is Auto Renewal':'Is_Auto_Renewal','Payment Type':'Payment_Type'},inplace=True)
        df=df[(pd.isnull(df['CnBio_ID'])== True)&((df.Is_Auto_Renewal==True)|(df.Payment_Type=='RT'))]
        df = df.ix[:,['imod_member_id','First Name','Last Name','CnBio_Birth_date']]
        df.rename(columns={'First Name':'First_Name','Last Name':'Last_Name'},inplace=True)
        df.drop_duplicates(inplace=True)
        df['First_Name']=pd.core.strings.str_strip(df['First_Name'])
        df['Last_Name']=pd.core.strings.str_strip(df['Last_Name'])
        merge=pd.merge(df,recon_data,how='left',on=['First_Name','Last_Name','CnBio_Birth_date'])
        merge['review']='Merge'
        merge=merge.ix[:,['CnBio_ID','imod_member_id','review','First_Name', 'Last_Name','CnBio_Birth_date']]
        merge.to_csv(export_path+'Merge_'+i,index=False)
        if len(merge.set_index(['First_Name','Last_Name']).index.get_duplicates()):
            print("Duplicate records found, Please check \n[('First_Name', 'Last_Name')]")
            print(merge.set_index(['First_Name','Last_Name']).index.get_duplicates())
