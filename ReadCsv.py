import pandas as pd
import glob
import os
import os
import glob
import pandas as pd
import re
from datetime import datetime as dt
os.chdir("C://Users//giorgos//Desktop//576013_1042828_compressed_COVID-19 Radiography Database (2)//")
import pandas as pd
import glob

path = "F://"

df1 = pd.read_csv(path + "mergednew12withoutnan.csv")

'''
print(df1.head())

df1=df1.rename(columns={'false.2':'TimeStamp'})
df1['TimeStamp'] = pd.to_datetime(df1["TimeStamp"], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
df1=df1.drop(columns=['Unnamed: 0','Unnamed: 0.1'])


df1.to_csv(path+"mergednew12withoutnan.csv")
'''
#df1['TimeStamp'] = pd.to_datetime(df1["TimeStamp"], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
date_time = pd.to_datetime(df1.pop('TimeStamp'), format='%Y-%m-%d %H:%M:%S')
print(df1.head())
print(df1.describe().transpose())
'''

# creating a blank series
Type_new1 = pd.Series([])
Type_new2 = pd.Series([])
Type_new3 = pd.Series([])
Type_new4 = pd.Series([])
Type_new5 = pd.Series([])
Type_new6 = pd.Series([])

index = []
a = 0
a1 = 0
a2 = 0
a3 = 0
a4 = 0
a5 = 0



# running a for loop and asigning some values to series
for ind in range(len(df1)):
    if df1['true.3'][ind] == 'PM 2.5':
        Type_new1[ind] = df1['false.3'][ind]
        index.append(ind)

    elif df1['true.3'][ind] == 'PM 1':
        a += 1

        Type_new2[index[a-1]] = df1['false.3'][ind]

    elif df1['true.3'][ind] == 'PM 10':
        a1 += 1
        Type_new3[index[a1-1]] = df1['false.3'][ind]

    elif df1['true.3'][ind] == 'Humidity':
        a2 += 1
        Type_new4[index[a2-1]] = df1['false.3'][ind]

    elif df1['true.3'][ind] == 'Temperature':
        a3 += 1
        Type_new5[index[a3-1]] = df1['false.3'][ind]

    elif df1['true.3'][ind] == 'TimeStamp':
        a4 += 1
        Type_new6[index[a4-1]] = df1['false.3'][ind]







    # inserting new column with values of list made above
df1.insert(10, "PM 2.5", Type_new1)
df1.insert(11, "PM 1", Type_new2)
df1.insert(12, "PM 10", Type_new3)
df1.insert(13, "Humidity", Type_new4)
df1.insert(14, "Temperature", Type_new5)
df1.insert(15, "TimeStamp", Type_new6)

# list output
df1.head()

cols_to_keep = ['false.2','PM 1','PM 2.5','PM 10','Humidity','Temperature','TimeStamp']
df1.dropna(how='all')
print(df1[cols_to_keep].head())


#https://stackabuse.com/tensorflow-2-0-solving-classification-and-regression-problems/



df1[cols_to_keep].to_csv("C://Users//giorgos//Desktop//mergednew12.csv")

'''