import pandas as pd
import glob
import os
import os
import glob
import pandas as pd
import re
os.chdir("C://Users//giorgos//Desktop//576013_1042828_compressed_COVID-19 Radiography Database (2)//")
import pandas as pd
import glob
path = "C://Users//giorgos//Desktop//576013_1042828_compressed_COVID-19 Radiography Database (2)//"

df1 = pd.read_csv(path+"Viral Pneumonia.matadata.csv", sep=';')
df2 = pd.read_csv(path+"COVID-19.metadata.csv", sep=';')
df3 = pd.read_csv(path+"NORMAL.metadata.csv", sep=';')




'''
print(pd.concat([df1, df2, df3]))
merged=pd.concat([df1, df2, df3])
merged.to_csv("merged.csv", index=False, encoding='utf-8')


df1= pd.read_csv(path+"merged.csv")
#df1['FILE NAME'] =  df1['FILE NAME']+'.png'
df1['target'] = ['viral_pneumonia' if re.findall("^Viral",holder) else ('Normal' if re.findall("^NORMAL",holder) else 'covid-19') for holder in df1['FILE NAME']]
df1.to_csv("mergednew.csv")
print(df1['FILE NAME'])

'''

covid_dir = "C://Users//giorgos//Desktop//576013_1042828_compressed_COVID-19 Radiography Database (2)//COVID-19//"
normal_dir = "C://Users//giorgos//Desktop//576013_1042828_compressed_COVID-19 Radiography Database (2)//NORMAL//"
pneumonia_dir = "C://Users//giorgos//Desktop//576013_1042828_compressed_COVID-19 Radiography Database (2)//Viral Pneumonia//"
for count, filename in enumerate(os.listdir(covid_dir)):
    dst = "COVID-19(" + str(count+1)+')' + " .png"
    src = covid_dir + filename
    dst = covid_dir + dst

    # rename() function will
    # rename all the files
    os.rename(src, dst)
