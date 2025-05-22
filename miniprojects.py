import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import random

# 讀取資料
df = pd.read_excel('miniproject.xlsx')

# y 變數：WeeksWithService
y = df['WeeksWithService']

# X 變數：去掉不相關欄位
drop_cols = ['CustID', 'ServiceStartDate', 'Credit','Age', 'AnnualIncome', 'WeeksWithService','Classification1','Classification2']
X_raw = df.drop(columns=drop_cols)

# 處理數據
# AnnualIncome 分類
bins = [0, 32004,62227, 130624, float('inf')]
labels = ['Low','mid', 'intermid', 'high']
df['Annual_'] = pd.cut(df['AnnualIncome'], bins=bins, labels=labels)

# Age 分類
bins = [0, 27, 32, 39, float('inf')]
labels = ['<27',' 27>.>32', ' 32>..>39', ' > 39']
df['Age_'] = pd.cut(df['Age'], bins=bins, labels=labels)

# Credit 分類
bins = [0, 27, 32, 39, float('inf')]
labels = ['Low','mid', 'intermid', 'high']
df['Credit_'] = pd.cut(df['Credit'], bins=bins, labels=labels)

# Classification1 分類
bins = [1, 2, 3, 4, float('inf')]
labels = ['1','2', '3', '4']
df['Classification1-'] = pd.cut(df['Classification1'], bins=bins, labels=labels)

# Classification2 分類
bins = [1, 2, 3, 4, float('inf')]
labels = ['1','2', '3', '4']
df['Classification2-'] = pd.cut(df['Classification2'], bins=bins, labels=labels)



X_raw=X_raw.join(df[['Credit_','Age_','Annual_','Classification1-','Classification2-']])


# One-hot encoding 全部轉成數值
X = pd.get_dummies(X_raw, drop_first=True).astype(float)

# 去掉缺失值
df_clean = pd.concat([X, y], axis=1).dropna()
X = df_clean.drop(columns=['WeeksWithService'])
y = df_clean['WeeksWithService']

# 線性回歸
X_with_const = sm.add_constant(X).astype(float) 
model = sm.OLS(y, X_with_const).fit()

print(model.summary())


summary_df = pd.DataFrame({
    'Feature': model.params.index,
    'Coefficient': model.params.values,
    'Std Error': model.bse,
    't-value': model.tvalues,
    'p-value': model.pvalues,
    'CI_lower': model.conf_int()[0],
    'CI_upper': model.conf_int()[1]
})

"""
summary_df= summary_df.round(4)
filename=(f"{random}.csv")
summary_df.to_csv(filename, index=False)
print("迴歸結果已匯出到 regression_results.csv")
"""

# 取得係數與 95% 信賴區間
conf_int = model.conf_int().loc[X.columns]
coefs = model.params.loc[X.columns]

# 整理係數與信賴區間
coef_df = model.params.loc[X.columns].reset_index()
coef_df.columns = ['Feature', 'Coefficient']
conf_int = model.conf_int().loc[X.columns]
coef_df['CI_lower'] = conf_int[0].values
coef_df['CI_upper'] = conf_int[1].values

# 只留下顯著（信賴區間不含0）的變數
sig_features = coef_df[(coef_df['CI_lower'] > 0) | (coef_df['CI_upper'] < 0)]
print(sig_features[['Feature', 'Coefficient']])

# 排序
coef_df = coef_df.sort_values('Coefficient')

# 畫棒棒糖圖
plt.figure(figsize=(10, len(coef_df) / 2))
sns.set_style("whitegrid")

# 顏色區分正負影響
palette = coef_df['Coefficient'].apply(lambda x: 'green' if x > 0 else 'red')



# 視覺化資料



# 畫棒棒糖
plt.hlines(y=coef_df['Feature'], xmin=coef_df['CI_lower'], xmax=coef_df['CI_upper'], color='black', alpha=0.7)
plt.scatter(coef_df['Coefficient'], coef_df['Feature'], color=palette)

plt.title('Feature Importance (Coefficient & 95% CI)', fontsize=14)
plt.xlabel('Coefficient')
plt.ylabel('')
plt.tight_layout()
plt.show()

# 只取分類變數 (示範以 ServiceType 開頭為例)
service_cols = [col for col in coef_df['Feature'] if 'ServiceType' in col]
subset_df = coef_df[coef_df['Feature'].isin(service_cols)]

plt.figure(figsize=(8, len(subset_df) * 0.6))
sns.barplot(x='Coefficient', y='Feature', data=subset_df,palette='viridis')

plt.title('ServiceType Effect on WeeksWithService', fontsize=14)
plt.xlabel('Coefficient')
plt.ylabel('')
plt.tight_layout()
plt.show()

# 畫annual:
# 只取分類變數 (示範以 ServiceType 開頭為例)
service_cols = [col for col in coef_df['Feature'] if 'Annual' in col]
subset_df = coef_df[coef_df['Feature'].isin(service_cols)]

plt.figure(figsize=(8, len(subset_df) * 0.6))
sns.barplot(x='Coefficient', y='Feature', data=subset_df, palette='colorblind')

plt.title('Annunal income Effect on WeeksWithService', fontsize=14)
plt.xlabel('Coefficient')
plt.ylabel('')
plt.tight_layout()
plt.show()

# 畫AGE:
# 只取分類變數 (示範以 ServiceType 開頭為例)
service_cols = [col for col in coef_df['Feature'] if 'Age' in col]
subset_df = coef_df[coef_df['Feature'].isin(service_cols)]

plt.figure(figsize=(8, len(subset_df) * 0.6))
sns.barplot(x='Coefficient', y='Feature', data=subset_df, palette='PuRd')

plt.title('AGE Effect on WeeksWithService', fontsize=14)
plt.xlabel('Coefficient')
plt.ylabel('')
plt.tight_layout()
plt.show()

# 畫Market:
# 只取分類變數 (示範以 ServiceType 開頭為例)
service_cols = [col for col in coef_df['Feature'] if 'Market' in col]
subset_df = coef_df[coef_df['Feature'].isin(service_cols)]

plt.figure(figsize=(8, len(subset_df) * 0.6))
sns.barplot(x='Coefficient', y='Feature', data=subset_df, palette='BuGn')

plt.title('Market Effect on WeeksWithService', fontsize=14)
plt.xlabel('Coefficient')
plt.ylabel('')
plt.tight_layout()
plt.show()

# 畫AGE:
# 只取分類變數 
service_cols = [col for col in coef_df['Feature'] if 'MaritalStatus' in col]
subset_df = coef_df[coef_df['Feature'].isin(service_cols)]

plt.figure(figsize=(8, len(subset_df) * 0.6))
sns.barplot(x='Coefficient', y='Feature', data=subset_df, palette='YlGn')

plt.title('MaritalStatus Effect on WeeksWithService', fontsize=14)
plt.xlabel('Coefficient')
plt.ylabel('')
plt.tight_layout()
plt.show()

# 畫Playment:
# 只取分類變數 
service_cols = [col for col in coef_df['Feature'] if 'PaymentMethod' in col]
subset_df = coef_df[coef_df['Feature'].isin(service_cols)]

plt.figure(figsize=(8, len(subset_df) * 0.6))
sns.barplot(x='Coefficient', y='Feature', data=subset_df, palette='Set1')

plt.title('PaymentMethod Effect on WeeksWithService', fontsize=14)
plt.xlabel('Coefficient')
plt.ylabel('')
plt.tight_layout()
plt.show()

# 畫Classification:
# 只取分類變數 
service_cols = [col for col in coef_df['Feature'] if 'Classification' in col]
subset_df = coef_df[coef_df['Feature'].isin(service_cols)]

plt.figure(figsize=(8, len(subset_df) * 0.6))
sns.barplot(x='Coefficient', y='Feature', data=subset_df, palette='Set1')

plt.title('Classification Effect on WeeksWithService', fontsize=14)
plt.xlabel('Coefficient')
plt.ylabel('')
plt.tight_layout()
plt.show()