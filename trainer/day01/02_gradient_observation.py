import torch
import torch.optim as optim
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化一个输入值
x = torch.tensor([[1.0]],device = device)

# 初始化对应的真实目标值 / 标签（target / label）
y = torch.tensor([[3.0]],device = device)

# 创建一个线性模型 y = weight * x + bias
# nn.Linear 会自动创建并初始化可训练模型参数 weight 和 bias
# 后续所谓的训练，其实就是不断的调整 weight 和 bias 值，使上面的公式计算结果无限趋近于我们提供的 y 值
model = nn.Linear(1,1).to(device)

# 定义损失函数，用于衡量模型预测值和真实值之间的误差程度
loss_fn = nn.MSELoss()

# 创建 GSD 优化器
# optimizer 后续会根据 gradient 和 learning rate 更新模型参数，其中 lr 就是learning rate，
# 本质上 lr 就是告诉 optimizer 每次沿着梯度给出的方向走多大一步，
# 这个值是人为选择和实验调优的参数，不是越大越好，也不是越小越好，
# 越大可能会导致 loss 计算后变成负数，越小可能需要训练很多次。
optimizer = optim.SGD(model.parameters(), lr=0.01)

print("=== 1. INITIAL ===")
print("weight:", model.weight.data)
print("bias:  ", model.bias.data)

print("weight.grad:", model.weight.grad)
print("bias.grad:  ", model.bias.grad)


# Forward
# 使用模型当前的 weight 和 bias 对输入的 x 进行计算
# 得到模型的预测结果 prediction 。本质上就是将 x 值代入上述的公式，得到一个 y
# 可以理解为 prediction = weight * x + bias
prediction = model(x)

print("\n=== 2. AFTER FORWARD ===")
print("prediction:", prediction.item())
print("weight:", model.weight.data)
print("bias:  ", model.bias.data)

print("weight.grad:", model.weight.grad)
print("bias.grad:  ", model.bias.grad)


# Loss
# 比较模型预测值 prediction 和真实目标值 y，
# 计算当前预测的损失度（也就是误差程度）
# 这里用到的是 MSELoss 本质上就是计算 (prediction - y)²
loss = loss_fn(prediction, y)

print("\n=== 3. AFTER LOSS ===")
print("loss:", loss.item())


# Backward
#
# 从 loss 开始反向传播，
# 计算 loss 对每一个可训练参数的梯度（这里涉及到高数，不解释了），
# 结果写入 parameter.grad
#
# 注意：这里只计算梯度，不修改模型参数
loss.backward()

print("\n=== 4. AFTER BACKWARD ===")
print("weight:", model.weight.data)
print("bias:  ", model.bias.data)

print("weight.grad:", model.weight.grad)
print("bias.grad:  ", model.bias.grad)


# Optimizer
# 根据 backward 计算出的梯度以及 learning 让特，
# 真正更新模型的 weight 、 bias 等可训练值
# 本次我们初始化的 optimizer 是通过 GSD 方法得到的，本质上是：
# parameter_new = parameter_old - learning_rate * gradient
optimizer.step()

print("\n=== 5. AFTER OPTIMIZER.STEP ===")
print("weight:", model.weight.data)
print("bias:  ", model.bias.data)

print("weight.grad:", model.weight.grad)
print("bias.grad:  ", model.bias.grad)
