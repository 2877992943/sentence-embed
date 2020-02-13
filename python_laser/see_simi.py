

import pandas as pdd
embedding=pdd.read_pickle('emb.pkl') #[3,dim]
import numpy as np

simi=np.dot(embedding,np.transpose(embedding))
print (simi)

##### 分词
# [[0.16885713 0.12882559 0.07686459]
#  [0.12882559 0.20230894 0.07115521]
#  [0.07686459 0.07115521 0.189074  ]]

###### 字做输入
# [[0.18728831 0.11024997 0.07532479]
#  [0.11024997 0.14469035 0.07433288]
#  [0.07532479 0.07433288 0.15966755]]