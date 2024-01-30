import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

df=yf.download('SPY').reset_index()

for m in [10,20,30,50,100]:
    df[f'feat_dist_from_ma_{m}']=df['Close']/df['Close'].rolling(m).mean()-1

for m in [3,5,10,15,20,30,50,100]:
    df[f'feat_dist_from_max_{m}']=df['Close']/df['High'].rolling(m).max()-1
    df[f'feat_dist_from_min_{m}']=df['Close']/df['Low'].rolling(m).min()-1

for m in [1,2,3,4,5,10,15,20,30,50,100]:
    df[f'feat_price_dist_{m}']=df['Close']/df['Close'].shift(m)-1

df['target_ma']=df['Close'].rolling(20).mean()
df['price_above_ma']=df['Close']>df['target_ma']
df['target']=df['price_above_ma'].astype(int).shift(-5)

df=df.dropna()

feat_cols=[col for col in df.columns if 'feat' in col]
train_until='2019-01-01'

x_train=df[df['Date']<train_until][feat_cols]
y_train=df[df['Date']<train_until]['target']

x_test=df[df['Date']>=train_until][feat_cols]
y_test=df[df['Date']>=train_until]['target']

clf=RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42,
    class_weight='balanced',
)

clf.fit(x_train,y_train)

y_train_pred=clf.predict(x_train)
y_test_pred=clf.predict(x_test)

train_accuracy=accuracy_score(y_train,y_train_pred)
train_precision=precision_score(y_train,y_train_pred)

test_accuracy=accuracy_score(y_test,y_test_pred)
test_precision=precision_score(y_test,y_test_pred)

print(test_accuracy)
print(test_precision)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 
import seaborn as sns

def plot_confusion_matrix(y_true, Y_pred, title, normalize):

    if normalize:
        cm = confusion_matrix(y_true, Y_pred, normalize=' pred' )
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues')
    else:
        cm = confusion_matrix(y_true, Y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    return

plot_confusion_matrix(y_train, y_train_pred, title='Training Data', normalize=False)
plot_confusion_matrix(y_train, y_train_pred, title='Training Data - Normalized', normalize=True)

plot_confusion_matrix(y_test, y_test_pred, title='Testing Data', normalize=False)
plot_confusion_matrix(y_test, y_test_pred, title='Testing Data - Normalized', normalize=True)