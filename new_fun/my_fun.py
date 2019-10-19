import torch 
import numpy as np
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Function
import torch.nn.functional as F

def get_orthogonal(rows,cols):
    re=np.zeros((rows,cols),dtype=np.float32)
    step=1/(rows//cols+2)
    re1=np.zeros((rows),dtype=np.float32)
    for i in range(rows):
        re[i][i%cols]=1
        re1[i]=-step*(i//cols+1)
    
    return re,re1


class Relu(Function):
    def forward(self,inp):
        self.save_for_backward(inp)
        output=inp.clamp(min=0)
        return output
    def backward(self,grad_output):
        inp=self.saved_tensors[0].view(-1)
        input_grad=grad_output.clone()
        input_grad1=input_grad.view(-1)
        for i in range(0,torch.numel(input_grad1),1):
            if(inp[i]<0):
                input_grad1[i]*=0.1
        return input_grad;
class URelu(Function):
    def forward(self,inp):
        self.save_for_backward(inp)
        output=inp.clamp(min=0,max=1)
        return output
    def backward(self,grad_output):
        inp=self.saved_tensors[0].view(-1)
        input_grad=grad_output.clone()
        input_grad1=input_grad.view(-1)
        for i in range(0,torch.numel(input_grad1),1):
            if(inp[i]>1):
                input_grad1[i]*=0.1/inp[i]
            elif(inp[i]<0):
                input_grad1[i]*=0.1/(1-inp[i])
        return input_grad

class Linear(nn.Module):
    def __init__(self,out_features,in_features,bias=True):
        super(Linear,self).__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.weight=Parameter(torch.Tensor(out_features,in_features))
        if(bias):
            self.bias=Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias',None)
        self.reset_parameters()
    def reset_parameters(self):
        #self.weight.data=torch.zeros_like(self.weight.data)
        """lenmin=0;
        if(self.in_features<self.out_features):
            lenmin=self.in_features;
        else:
            lenmin=self.out_features;
        for i in range(0,lenmin,1):
            self.weight.data[i][i]=1"""
        re,re1=get_orthogonal(self.out_features,self.in_features)
        self.weight.data=torch.from_numpy(re)
        self.bias.data=torch.from_numpy(re1)
        re1=torch.from_numpy(re1)
        
        #print("Linear:   ",self.weight.data,self.bias.data)

        
    def forward(self,inp):
        return F.linear(inp,self.weight,self.bias)
class LiURelu(nn.Module):
    def __init__(self,shape):
        super(LiURelu,self).__init__()
        self.mul=Parameter(torch.ones(shape))
        self.bias=Parameter(torch.zeros(shape))
    def forward(self,inp):
        ure=URelu()
        return ure(inp*self.mul+self.bias)


def init_convkernel(conv2):
    #r=torch.zeros_like(k.weight.data)
    cols=1
    rows=conv2.weight.data.shape[0]
    temp_m=conv2.weight.data.view(rows,-1)
    cols=temp_m.shape[1]
    re,re1=get_orthogonal(rows,cols)
    re=torch.from_numpy(re)
    re1=torch.from_numpy(re1)
    for i in range(rows):
        for j in range(cols):
            temp_m[i][j]=re[i][j]
        conv2.bias.data[i]=re1[i]
    #print(conv2.weight.data,conv2.bias.data)
    #for i in range(conv2.shape[0]):

        

def cite_param(net1,net2):
    for f in net1.__dict__["_modules"]:
        getattr(net2,f).weight=getattr(net1,f).weight
        getattr(net2,f).bias=getattr(net1,f).bias 

        #setattr(net2,f,getattr(net1,f))
def comput_grad_norm(param):
    temp_sum=0;temp_sum1=0;
    for f in param:
        temp_sum+=(f.grad.data**2).sum()
        temp_sum1+=f.grad.data.norm()**2
    print("grad norm: ",temp_sum,temp_sum1)
    return temp_sum


class My_opt():
    def __init__(self,rate,param):
        self.loss=0.0;
        self.rate=rate
        self.param=param
        self.rate1=1.0

    def optim(self,loss):
        if self.loss==loss:
            self.rate1=self.rate1*1.2+0.1
        else:
            self.rate1=1.0
        self.loss=loss
        norm=comput_grad_norm(self.param)
        for f in self.param:
            f.data.sub_(f.grad.data/norm*loss*self.rate*self.rate1)


