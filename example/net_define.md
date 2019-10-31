## 自定义实现模型

- 线性模型

~~~
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)
    # forward 定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y
    
net = LinearNet(num_inputs)
print(net) # 使用print可以打印出网络的结构

# 输出
LinearNet(
  (linear): Linear(in_features=2, out_features=1, bias=True)
)
~~~

## 搭建网络

~~~
# 写法一
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
    # 此处还可以传入其他层
    )

# 写法二
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))
# net.add_module ......

# 写法三
from collections import OrderedDict
net = nn.Sequential(OrderedDict([
          ('linear', nn.Linear(num_inputs, 1))
          # ......
        ]))

print(net)
print(net[0])

# 输出
Sequential(
  (linear): Linear(in_features=2, out_features=1, bias=True)
)
Linear(in_features=2, out_features=1, bias=True)
~~~

## 参数初始化

~~~
from torch.nn import init

# 将权重参数每个元素初始化为随机采样于均值为0、标准差为0.01的正态分布
init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0)  # 也可以直接修改bias的data: net[0].bias.data.fill_(0)

init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0) 
~~~

## 定义损失函数

~~~
loss = nn.MSELoss()
~~~

## 定义优化算法

- `torch.optim`模块提供了很多常用的优化算法比如SGD、Adam和RMSProp等

~~~
import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)
~~~

- 为不同子网络设置不同的学习率，这在finetune时经常用到

~~~
optimizer =optim.SGD([
                # 如果对某个参数不指定学习率，就使用最外层的默认学习率
                {'params': net.subnet1.parameters()}, # lr=0.03
                {'params': net.subnet2.parameters(), 'lr': 0.01}
            ], lr=0.03)
~~~

- 调整学习率

~~~
# 调整学习率
for param_group in optimizer.param_groups:
    param_group['lr'] *= 0.1 # 学习率为之前的0.1倍
~~~

## 训练模型

- 按照小批量随机梯度下降的定义，我们在`step`函数中指明批量大小，从而对批量中样本梯度求平均

~~~
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))  # 计算loss
        optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
        l.backward()  # 反向传播
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))
~~~

