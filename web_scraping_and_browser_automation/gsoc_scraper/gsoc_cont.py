import re
import json
import pandas as pd


#loading json data
f=open('students.json','r')
file=f.read()
jso=json.loads(file)
#print(len(jso))

#loading csv data
df=pd.read_csv('project_data.csv')
names_csv=df.iloc[:,0]
n_csv=names_csv.values.reshape(-1,1)


#checking official names
def single(name):
    if(len(name.split(' '))>=2):
        return True
    else:
        return False
def special(name):
    regex=re.compile('[@_!#$%^&*()<>?/\|}{~:0-9]')
    if(regex.search(name)==None):
        return True
    else:
        return False

def isOfficial(name):
    if(special(name) and single(name) and not(name.islower())):
        return True
    else:
        return False
#creating a list to store official names
names_json=[]
#function to extract proper names
def official_name():
#print(jso.get('n'))
    for i in range(len(jso)):
        name=jso[i]['n']
        if(isOfficial(name)):
           names_json.append(name)
        else:
            print(name)
            


#calling the function to official names
official_name()

#print(len(jso))
#print("Lavanya Singh" in n_csv)


for i,nm_csv in enumerate(names_csv):
    for j in range(len(jso)):
        if(jso[j]["n"]==nm_csv and isOfficial(nm_csv)):
            print("Name:",nm_csv," Roll No:",jso[j]["i"]," Branch:",jso[j]["d"]," ",df.iloc[i,2]," Project:",df.iloc[i,1])
        #print()
        #print(name)
#print(len(names_json))
