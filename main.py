import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# データの読み込み
train_deta = pd.read_csv('/Users/g/Desktop/kaggle/v_1.0/train_data.csv')
test_deta = pd.read_csv('/Users/g/Desktop/kaggle/v_1.0/test_data.csv')

# 必要なカラムの選択
df = train_deta[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

# 特徴量とターゲット変数に分ける
X = df.drop('Survived', axis=1)
y = df['Survived']

# 学習用データの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# データの標準化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ロジスティック回帰モデルの作成と学習
model = LogisticRegression()
model.fit(X_train, y_train)

# モデルの評価
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# テストデータ(test_deta)の前処理と予測
X_submission = test_deta[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

# カテゴリデータの数値変換が必要な場合
# 例： 'Sex' 列を数値に変換 (例: {"male": 0, "female": 1})
X_submission['Sex'] = X_submission['Sex'].map({"male": 0, "female": 1})

# 欠損値を補完
X_submission['Age'].fillna(X_submission['Age'].mean(), inplace=True)
X_submission['Fare'].fillna(X_submission['Fare'].mean(), inplace=True)

# 標準化
X_submission = scaler.transform(X_submission)

# 予測
y_submission = model.predict(X_submission)

# 結果を保存
output = pd.DataFrame({'PassengerId': test_deta['PassengerId'], 'Survived': y_submission})
output.to_csv('titanic_predictions.csv', index=False)
