from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as npy
import csv






dt = []
closingPrice = []



def PredClose(dt, closingPrice, y):
	dt = npy.reshape(dt,(len(dt), 1)) # converting to matrix of n X 1

	svr_lin = SVR(kernel= 'linear', C= 1e3)
	svr_poly = SVR(kernel= 'poly', C= 1e3, degree= 2)
	svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1) # defining the support vector regression models
	svr_rbf.fit(dt, closingPrice) # fitting the data points in the models
	svr_lin.fit(dt, closingPrice)
	svr_poly.fit(dt, closingPrice)

	plt.scatter(dt, closingPrice, color= 'black', label= 'Data') # plotting the initial datapoints 
	plt.plot(dt, svr_rbf.predict(dt), color= 'red', label= 'RBF') # plotting the line made by the RBF kernel
	plt.plot(dt,svr_lin.predict(dt), color= 'green', label= 'Linear') # plotting the line made by linear kernel
	plt.plot(dt,svr_poly.predict(dt), color= 'blue', label= 'Polynomial') # plotting the line made by polynomial kernel
	plt.xlabel('dt')
	plt.ylabel('closingPrice')
	plt.title('Support Vector Regression')
	plt.legend()
	plt.show()

	return svr_rbf.predict(y)[0], svr_lin.predict(y)[0], svr_poly.predict(y)[0]

def get_data(fn):
	with open(fn, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)	# skipping column names
		for row in csvFileReader:
			dt.append(int(row[0].split('-')[0]))
			closingPrice.append(float(row[1]))
	return

get_data('amzn.csv') # calling get_data method by passing the csv file to it
#print "dt- ", dt
#print "closingPrice- ", closingPrice

predicted_closingPrice = PredClose(dt, closingPrice, 30)
