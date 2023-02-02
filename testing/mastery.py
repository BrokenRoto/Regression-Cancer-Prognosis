import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# para la primera predicción
df = pd.read_csv('../dataset.csv')

# borrar la ultima columna
df.pop(df.columns[-1])

# drop rows with missing values
df.dropna(axis=0, inplace=True)

dataset = df.values
# split into input (X) and output (Y) variables
X = dataset[:,0:70]
Y = dataset[:,70]

# Y = df['TIMEsurvival_Years']
# X = df.drop('TIMEsurvival_Years', axis=1)
#
# x = np.asarray(X).astype('float32')
# Y = np.asarray(Y).astype('float32')


# define base model
# def baseline_model():
#     # create model
#     model = Sequential()
#     model.add(Dense(70, input_shape=(70,), kernel_initializer='normal', activation='relu'))
#     model.add(Dense(1, kernel_initializer='normal'))
#     # Compile model
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     return model
# # evaluate model with standardized dataset
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(model=baseline_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=10)
# results = cross_val_score(pipeline, X, Y, cv=kfold, scoring='neg_mean_squared_error')
# print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# define wider model
def wider_model():
    # create model
    model = Sequential()
    model.add(Dense(20, input_shape=(70,), kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
# evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(model=wider_model, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10)
results = cross_val_score(pipeline, X, Y, cv=kfold, scoring='neg_mean_squared_error')
print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# let's see the training and validation accuracy by epoch
history_dict = history.history
loss_values = history_dict['loss'] # you can change this
val_loss_values = history_dict['val_loss'] # you can also change this
epochs = range(1, len(loss_values) + 1) # range of X (no. of epochs)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'orange', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()