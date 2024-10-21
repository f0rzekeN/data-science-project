import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 读取数据
df = pd.read_csv('bike.csv')

# 删除无用的id列
df.drop(columns=['id'], inplace=True)

# 提取目标变量y并删除y列
y = df['y'].values.reshape(-1, 1)
df.drop(columns=['y'], inplace=True)

# 转换为Numpy数组
X = df.values

# 将数据划分为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler_x = StandardScaler()
scaler_y = StandardScaler()
x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.transform(x_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# 构建线性回归模型并训练
model = LinearRegression()
model.fit(x_train, y_train)

# 使用模型预测
y_pred = model.predict(x_test)

# 将预测结果和测试标签反归一化
y_pred = scaler_y.inverse_transform(y_pred)
y_test = scaler_y.inverse_transform(y_test)

# 计算RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"模型的RMSE值为: {rmse}")