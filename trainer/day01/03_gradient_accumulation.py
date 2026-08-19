# 梯度累加代码演示
import torch
import torch.nn as nn
import torch.optim as optim

# 这里加载英伟达 cuda 相关库，利用英伟达 GPU 计算，如果加载失败则使用 CPU 初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义输入值和正确答案
x = torch.tensor([[1.0]],device = device)
y = torch.tensor([[3.0]],device = device)

# 为了方便观察，我们手工固定初始化参数
model = nn.Linear(1,1).to(device)

with torch.no_grad():
    model.weight.fill_(1.0)
    model.bias.fill_(1.0)

loss_fn = nn.MSELoss()

optimizer = optim.SGD(model.parameters(),lr=0.01)

def print_grad(title):
    print(f"\n === {title} ===")
    print("weight:", model.weight.data)
    print("bias:", model.bias.data)
    print("weight.grad",model.weight.grad)
    print("bias.grad",model.bias.grad)

print_grad("INSTALL")


# ==========================================
# 第一次 backward
# ==========================================

prediction = model(x)
loss = loss_fn(prediction,y)

print("\nFirst_prediction:",prediction.item())
print("First loss:", loss.item())

loss.backward()

print_grad("AFTER FIRST BACKWARD")

# ==========================================
# 第二次 backward
#
# 注意：
# 这里故意不执行 optimizer.zero_grad()
# ==========================================

prediction = model(x)
loss = loss_fn(prediction, y)

print("\nSecond prediction:", prediction.item())
print("Second loss:", loss.item())

loss.backward()

print_grad("AFTER SECOND BACKWARD WITHOUT ZERO_GRAD")


# ==========================================
# 清空 Gradient
# ==========================================

optimizer.zero_grad()

print_grad("AFTER ZERO_GRAD")


# ==========================================
# 第三次 backward
# ==========================================

prediction = model(x)
loss = loss_fn(prediction, y)

loss.backward()

print_grad("AFTER THIRD BACKWARD")


# ==========================================
# 真正修改模型参数
# ==========================================

optimizer.step()

print_grad("AFTER OPTIMIZER.STEP")













