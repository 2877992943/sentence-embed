import pandas as pdd

from sklearn.cluster import KMeans
import numpy as np
import json


########## 看两个东西
# 1，语义聚类效果   是不是相似语义的句子在一起
# 2，摘要 聚类中心  能否代替文章

def calc_dis(v1,v2):
    #return np.sum((v1-v2)**2)
    return np.dot(v1,v2)


ll=pdd.read_pickle('./all.pkl')

inp=[]
arrll=[]

for batch in ll[:]:
    titles,arr=batch
    arrll+=[ar for ar in arr]
    inp.extend(titles)
    #print ('')


vec=np.array(arrll)#[batch step dim]

#####用什么方法 去掉STEP 维度
#vec=np.mean(vec,axis=1)
#vec=vec.reshape([-1,vec.shape[1]*vec.shape[2]])
#vec=vec[:,0,:]


n_cluster=100
est=KMeans(init='k-means++', n_clusters=n_cluster, n_init=100)
est.fit(vec)
lab=est.labels_ #[n,]
center=est.cluster_centers_ #[10cluster,dim]


### 每个LAB 对应STR
cluster_input={}
for xind in range(len(lab)):
    l=lab[xind]
    x=inp[xind]
    vec_i=vec[xind]
    if l in cluster_input:
        cluster_input[l].append([x,vec_i])
    if l not in cluster_input:
        cluster_input[l]=[[x,vec_i]]
    ####

##### 每个cluster  分别对 member 距离排序
c_ll={}
for c,vll in cluster_input.items():
    c_ll[c]=[]
    ## each cluster
    center_i=center[c] #[dim,]
    x_dis=[[p[0],calc_dis(center_i,p[1])] for p in vll]
    ll=sorted(x_dis,key=lambda s:s[1],reverse=True)
    ####
    c_ll[c]=ll

######
#writer=open('rst.json','w')
rst=[]
for c,ll in c_ll.items():
    #ll=[[l[0],float(l[1])] for l in ll]
    for text,dis in ll:
        rst.append([c,text,dis])

    # writer.write(json.dumps({str(c):ll},ensure_ascii=False,
    #                         #indent=4
    #                         ))

#####
df=pdd.DataFrame({'c':[p[0] for p in rst],
               'text':[p[1] for p in rst],
                'dis':[p[2] for p in rst]})


from problem_util_yr.loadDict.csv2xls import writeXLS_fromDF_toLocal
writeXLS_fromDF_toLocal(df,'cluster_center')



