import torch
from numpy.ma.extras import average
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm


def train_epoch(model,train_loader,criterion,optimizer,device='cuda'):
    """训练一个epoch"""
    # 先将模型设置为训练模式
    # 启用 Dropout、BatchNorm 等层的训练行为，与 model.eval() 相对
    model.train()
    # 定义总训练损失
    total_loss = 0
    # 存储所有批次的预测结果
    all_preds = []
    # 存储所有批次的真实标签
    all_labels = []

    # 遍历训练数据加载器，每次获取一个批次的数据
    # desc="Training" 显示带有"Training"描述的进度条
    # batch_X_mark 来自于provider_6_2中的__getitem__函数
    for batch_X,batch_X_mark,batch_y in tqdm(train_loader,desc="Training"):
        # 将数据送入gpu中，否则无法训练
        batch_X = batch_X.to(device)
        batch_X_mark = batch_X_mark.to(device)
        # 将标签移动到设备并去除多余维度
        # squeeze() 将形状从 [batch_size, 1] 变为 [batch_size]
        batch_y = batch_y.squeeze().to(device)


        #PyTorch 默认累积梯度，每次反向传播前必须清零
        optimizer.zero_grad()
        # 生成这一批次的模型预测标签
        output = model(batch_X, batch_X_mark, None, None) # (batch_size,num_class)
        # 计算训练损失
        loss = criterion(output,batch_y)

        # 反向传播阶段
        loss.backward()
        # 根据计算出的梯度更新模型参数
        optimizer.step()

        # 将当前批次损失加入到总损失中
        # loss.item()将张量转换为Python标量值
        total_loss += loss.item()

        # 获取预测的标签
        # 按照类别维度取得最大值的索引
        preds = torch.argmax(output,dim=1).cpu().numpy()
        # 将preds中的元素逐个加入到all_preds中
        all_preds.extend(preds)
        # 将真实标签也加入到all_labels中
        all_labels.extend(batch_y.cpu().numpy())

    # 计算平均损失
    avg_loss = total_loss / len(train_loader)
    # 计算准确率
    accuracy = accuracy_score(all_labels,all_preds)

    return avg_loss, accuracy

def val_epoch(model,val_loader,criterion,device='cuda'):
    """验证一个epoch"""
    # 将模型设置为评估模式
    model.eval()
    # 同样初始化累积变量
    total_loss = 0
    all_preds = []
    all_labels = []

    # 禁用梯度计算上下文
    with torch.no_grad():
        # 对val_dataloader进行验证
        for batch_X,batch_X_mark,batch_y in val_loader:
            # 同样将数据加载到gpu上
            batch_X = batch_X.to(device)
            batch_X_mark = batch_X_mark.to(device)
            batch_y = batch_y.squeeze().to(device)

            output = model(batch_X, batch_X_mark, None, None)
            loss = criterion(output,batch_y)
            total_loss += loss.item()

            preds = torch.argmax(output,dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.cpu().numpy())

        # 计算平均损失
        avg_loss = total_loss / len(val_loader)

        # 计算准确率
        accuracy = accuracy_score(all_labels,all_preds)
        # 计算精确率 average='weighted'：按类别样本数加权平均
        #           zero_division=0：处理除零情况（当某类别无预测样本时）
        precision = precision_score(all_labels,all_preds)

        # average='weighted'
        recall = recall_score(all_labels,all_preds,average='weighted',zero_division=0)

        f1 = f1_score(all_labels,all_preds,average='weighted',zero_division=0)

        return avg_loss,accuracy,precision,recall,f1