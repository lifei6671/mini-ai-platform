import torch
import torch.nn as nn
import  torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device:", device)

# --------------------------------------------------
# 1. 准备训练数据
#
# 我们人为制造一个非常简单的规律：
#
# y = 2x + 1
#
# 目标是让神经网络自己学出：
# weight ≈ 2
# bias   ≈ 1
# --------------------------------------------------

x = torch.tensor(
    [[1.0],[2.0],[3.0],[4.0]],
    device = device
)

y = torch.tensor(
    [[3.0],[5.0],[7.0],[9.0]],
    device = device
)


# --------------------------------------------------
# 2. 创建模型
#
# y = wx + b
# --------------------------------------------------

model = nn.Linear(1,1).to(device)

# --------------------------------------------------
# 3. Loss
# --------------------------------------------------

loss_fn = nn.MSELoss()

# --------------------------------------------------
# 4. Optimizer
# --------------------------------------------------

optimizer = optim.SGD(
    model.parameters(),
    lr = 0.01,
)

print("\n=== BEFORE TRAINING ===")

print("weight:", model.weight.data)
print("bias:", model.bias.data)

# --------------------------------------------------
# 5. Training Loop
# --------------------------------------------------

for step in range(1000):
    # Forward
    prediction = model(x)

    # Loss
    loss = loss_fn(prediction, y)

    # 清除上一轮梯度
    optimizer.zero_grad()

    # Backward
    loss.backward()

    # 更新模型参数
    optimizer.step()

    if step % 100 == 0:
        print(
            f"step={step:4d}",
            f"loss={loss.item():.6f}"
        )


print("\n=== AFTER TRAINING ===")

print("weight:", model.weight.data)
print("bias:", model.bias.data)

# --------------------------------------------------
# 6. 测试
# --------------------------------------------------

test_x = torch.tensor([[10.0]], device=device)

with torch.no_grad():
    test_y = model(test_x)

print("\n10 ->", test_y.item())






















