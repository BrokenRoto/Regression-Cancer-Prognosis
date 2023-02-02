# MSE = cross_val_score(estimator, X, y, cv=cv, scoring='neg_mean_squared_error')
# RMSE = cross_val_score(estimator, X, y, cv=cv, scoring='neg_root_mean_squared_error')
# MAE = cross_val_score(estimator, X, y, cv=cv, scoring='neg_mean_absolute_error')

# for train_index, test_index in cv.split(X, y):
#     # data split
#     train = normalized_data.loc[train_index, :]
#     test = normalized_data.loc[test_index, :]
#     X_train = train.drop('TIMEsurvival_Years', axis=1)
#     y_train = train['TIMEsurvival_Years']
#     X_test = test.drop('TIMEsurvival_Years', axis=1)
#     y_test = test['TIMEsurvival_Years']
#
#     # create ANN model
#     model = Sequential()
#
#     # Defining the Input layer and FIRST hidden layer, both are same!
#     model.add(Dense(units=5, input_dim=70, kernel_initializer='normal', activation='relu'))
#
#     # Defining the Second layer of the model
#     # after the first layer we don't have to specify input_dim as keras configure it automatically
#     model.add(Dense(units=5, kernel_initializer='normal', activation='tanh'))
#
#     # The output neuron is a single fully connected node
#     # Since we will be predicting a single number
#     model.add(Dense(1, kernel_initializer='normal'))
#
#     # Compiling the model
#     model.compile(loss='mean_squared_error', optimizer='adam')
#
#     # Fitting the ANN to the Training set
#     model.fit(X_train, y_train, batch_size=20, epochs=50, verbose=1)
#
#
#     model.add(Dense(70, input_shape=(70,), kernel_initializer='normal', activation='relu'))
#     model.add(Dense(1, kernel_initializer='normal'))
#
#     model.compile(loss='mean_squared_error', optimizer='adam')
#
#
#     # prediction = model.predict(X_test)
#     #
#     # scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
#
#     # There are three error metrics that are commonly used for evaluating and reporting the performance of a regression model; they are:
#     # Mean Squared Error (MSE).
#     # Root Mean Squared Error (RMSE).
#     # Mean Absolute Error (MAE)