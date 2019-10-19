import numpy as np
import torch
def get_orthogonal(rows,cols):
    re=np.zeros((rows,cols),dtype=np.float32)
    step=1/(rows//cols+2)
    re1=np.zeros((rows),dtype=np.float32)
    for i in range(rows):
        re[i][i%cols]=1
        re1[i]=step*(i//cols+1)
    
    return re,re1


a,b=get_orthogonal(10,4)
print(a,b)


