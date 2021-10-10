t_raw_data = pd.read_csv('https://raw.githubusercontent.com/stutisehgal/MachineLearning/0d077bf91dcade8ecba67d7a2c0789f48cc15537/Multiple%20Linear%20Regression/chennai_house_multivariate_test.csv')
t_raw_data.head()
t_data=(t_raw_data-raw_data.mean())/(raw_data.max()-raw_data.min())
t_data.insert(0, 'Ones', 1)
t_data.head()
t_cols = t_data.shape[1]
print (t_cols)
t_x=t_data.iloc[:,0:t_cols-1]
t_y=t_data.iloc[:,t_cols-1:t_cols]
t_x.shape
t_x = np.matrix(t_x)
t_y = np.matrix(t_y)
Model_testdata_price =  t_x*new_theta.T
test_error = [np.power((b-a),2) for (a, b) in zip(Model_testdata_price, t_y)] #mean absolute percentage error
error = np.sum(test_error)

error=(error/len(y))*100

print("test error % = {}".format(error))
accuracy= 100 - error
print("test accuracy %={}".format(accuracy))
