from new_fun import *
#import new_fun
#from my_fun import *
a=np.random.randn(3,3)
data1=unpickle("./data/cifar-10-batches-py/data_batch_1")
image,label=get_batch(data1,99)

#show_image(image)
image=image.astype(np.float32)
image=image/255.0

image=torch.from_numpy(image)
image=image.permute(0,3,1,2)
print(image.shape)
label=label.astype(np.float32)
label=torch.from_numpy(label)/9.0
print("label: ",label)
#print(image)
#re=my_fun.Relu()
#image1=re(image)
#print(image1)
"""
inp=torch.tensor([[1.0],[2.0]])
label0=torch.tensor([3.6,4.8])

criterion=nn.MSELoss()
class Net0(nn.Module):
    def __init__(self):
        super(Net0,self).__init__()
        self.f1=my_fun.Linear(1,1)
    def forward(self,inp):
        re=my_fun.Relu()
        inp1=re(self.f1(inp))
        return inp1;
net0=Net0()
out=net0(inp)
print(inp)
rate=0.0002
param=list(net0.parameters())
print("param:",len(param))
for i in range(1000):
    print("inp:",inp)
    net0.zero_grad()
    out=net0(inp).view(-1)
    print("out: ",out)
    loss=criterion(out,label0)

    print("loss: ",loss)
    loss.backward()
    for f in param:
        print("f.data: ",f.data)
        print("f.grad.data: ",f.grad.data)
        f.data.sub_(f.grad.data*rate)
"""
    #k=torch.nn.Parameter(r)


class Net1(nn.Module):
    def __init__(self):
        super(Net1,self).__init__()
        self.conv1=nn.Conv2d(3,6,3,padding=2-1)
        self.conv2=nn.Conv2d(6,1,3,padding=2-1)
        #self.re1=my_fun.Relu()
        #self.re2=my_fun.Relu()
        init.constant_(self.conv1.weight.data,0.0)
        init.constant_(self.conv2.weight.data,0.0)
        #init.constant_(self.conv1.weight.data,0.1)
        #self.conv1.weight.data[]

        #init.constant_(self.conv2.weight.data,0.1)
        my_fun.init_convkernel(self.conv1)
        my_fun.init_convkernel(self.conv2)
        #print("conv1:  ",self.conv1.weight.data,self.conv1.bias.data)
        #print("conv2:  ",self.conv2.bias.data)

    def forward(self,inp):
        #re=my_fun.Relu()
        #ure=my_fun.Relu()
    
        inp2=my_fun.Relu()(self.conv1(inp))

        #print("out: net1 inp2: ",inp2)
        inp3=my_fun.Relu()(self.conv2(inp2))

        return inp3
class Net2(nn.Module):
    def __init__(self):
        super(Net2,self).__init__()
        self.f1=my_fun.Linear(32*32*2,32*32)
        self.f2=my_fun.Linear(1,32*32*2)
        #self.re1=my_fun.Relu()
        #self.ure1=my_fun.URelu()
     #   a=np.zeros((3,32,32,1),dtype=np.float32)
        
        self.a=torch.zeros(3,32,32,1)
        self.a[0][0][0][0]=1.0
    def forward(self,inp):
        #re=my_fun.Relu()
        #re2=my_fun.URel
        inp1=inp.view(inp.shape[0],-1)
        inp2=my_fun.Relu()(self.f1(inp1))
        #print("inp2: ",inp2.shape,inp2)
        inp3= my_fun.URelu()(self.f2(inp2))
          
        print("out: ",inp3)
        return self.a.matmul(inp3.view(1,-1)).permute(3,0,1,2)
def show_out(out):
    print("out:")
    temp_sum=0;
    for i in range(10):
        print(out[i][0][0][0])
        temp_sum+=(i/9.0-out[i][0][0][0])**2
    temp_sum=temp_sum/(1024*3)
    print("loss ",temp_sum)

 
net1=Net1()
net2=Net2()
rate=0.04
criterion=nn.MSELoss()
#param=list(net1.parameters())+list(net2.parameters())
#out1=net2(net1(image)).view(-1)
#print(out1.shape)

param=list(net1.parameters())+list(net2.parameters())
#print(net1.conv1.weight.data)
#for f in param:
   # print("f:shape:",f.data)
#out1=net2(net1(image)).view(-1)
#print(out1.shape)
"""for j in range(100):
    image,label=get_batch(data1,99-j)
    image=image.astype(np.float32)
    label=label.astype(np.float32)
    image=torch.from_numpy(image).permute(0,3,1,2)
    label=torch.from_numpy(label)
"""
fixed_p=torch.tensor([i for i in range(10)]).view(1,-1)
fixed_p=fixed_p.float()
a=torch.zeros(image.shape[1],image.shape[2],image.shape[3],1)
a[0][0][0][0]=1.0
fixed=a.matmul(fixed_p)/9.0
fixed=fixed.permute(3,0,1,2)
fixed=fixed.detach()
#print("yuan",fixed[1][0][0][0])


#print("param: ",param[1])
my_opt=my_fun.My_opt(rate,param)

for i in range(1000):
    net1.zero_grad()
    net2.zero_grad()

    out=net2(net1(fixed))
    #show_out(out)
    loss=criterion(out,fixed)
    #print("fixed:  ",fixed)
    loss.backward()
    #print(fixed.shape,"outshape:",out.shape)
    print("loss: %.20f" % loss)
    #print("grad",param[7].grad.data,param[6].grad.data)
    #print("data: ",param[7].data,param[6].data)
    param[7].data.sub_(param[7].grad.data*rate)
    param[6].data.sub_(param[6].grad.data*rate)
    my_opt.optim(loss)
    #norm=my_fun.comput_grad_norm(param)
    #print("rate1: ",rate1)
    #if loss1==loss:
        #rate1=rate1*1.2+0.1
    #else:
        #rate1=1.0
    #loss1=loss
    #for f in param:

        #print(f.grad.data)
        #f.data.sub_(f.grad.data/norm*loss*rate*rate1)


print("***********")
print("***********")
print("***********")
#print(label)

target=a.matmul(label.view(1,-1)).permute(3,0,1,2);
print("target: ",target.shape,image.shape)

for i in range(200):
    net1.zero_grad()
    net2.zero_grad()
    out=image
    for j in range(2):
        temp_net1=Net1()
        temp_net2=Net2()
        my_fun.cite_param(net1,temp_net1)
        my_fun.cite_param(net2,temp_net2)
        out=temp_net2(temp_net1(out))

        #print("out:",out)
    loss=criterion(out,target)
    loss.backward()
    print("loss: %.20f" % loss)

    my_opt.optim(loss);

torch.save(net1.state_dict(),"new_net1.pt")
torch.save(net2.state_dict(),"new_net2.pt")

    #loss=criterion)
""""
for j in range(1):
    image,label=get_batch(data1,999-2*j)
    image=image/255.0
    image=image.astype(np.float32)
    label=label.astype(np.float32)/9.0
    image=torch.from_numpy(image).permute(0,3,1,2)
    label=torch.from_numpy(label)

    for i in range(1000):
        net1.zero_grad()
        net2.zero_grad()
        out=net2(net1(image)).view(-1)
        loss=criterion(out,label)
        loss.backward()
        print("loss",loss,"out ",out,"label:",label)
        for f in param:
            f.data.sub_(f.grad.data*rate)
    
"""
"""
for i in range(300):
    image1=image.clone()
    for j in range(3):

        net1.zero_grad()
        net2.zero_grad()
        out=net2(net1(image1)).view(-1)
        loss=criterion(out,label)
        print("loss",loss)
        loss.backward()
        for f in param:
        #print(f.grad.data)
            f.data.sub_(f.grad.data*rate)
                #print(image1.shape)
        a=torch.ones(image.shape[1],image.shape[2],image.shape[3],1)
        image1=(a.matmul(out.view(1,-1)).permute(3,0,1,2)).detach()

out=net2(net1(image)).view(-1)
print(out,label)
"""

