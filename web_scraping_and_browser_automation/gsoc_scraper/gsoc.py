from bs4 import BeautifulSoup
import requests
import re
import csv
import pandas as pd
def striphtml(data):
    p = re.compile(r'<.*?>')
    return p.sub('', data)
d=[]
for k in range(1,13): 
    print(k,"page")
    url="https://summerofcode.withgoogle.com/archive/2019/projects/?page="+str(k)
    page=requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    #print(soup)
    a=soup.findAll('div',attrs={'class':'archive-project-card__header'})
    for elem in a:
        #print(a,"\n\n\n")
        b=striphtml(str(a))
        #print(b)
    c=b.split(',')
    #print(len(c))
    #print(c[108].split('\n'))
    for i in range(len(c)):
        #print(c[len(c)-1])
        temp=c[i].split('\n')
        if(len(temp)>=6):
            #print([temp[j] for j in (2,4,5)],"\n\n")
            d.append([temp[j] for j in (2,4,5)])
    print("success")
#e=[[1,2,3],[4,5,6]]
#df=pd.DataFrame(d)
#df.to_csv('out.csv',sep=',',header=None,index=None)
with open('project_data.csv','w',newline='') as fp:
    a=csv.writer(fp,delimiter=',')
    a.writerows(d)
    
print("data_successfully_scraped\nnow run the next script")
