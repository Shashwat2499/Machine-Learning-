import numpy as npm
import math as mt
import matplotlib.pyplot as pt

def main():
	print('START Q3_D\n')

	def clean_data(line):
		return line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')

	def fetch_data(filename):
		with open(filename, 'r') as f:
			input_data = f.readlines()
			clean_input = list(map(clean_data, input_data))
			f.close()
		return clean_input

	def readFile(dataset_path):
		input_data = fetch_data(dataset_path)
		input_np = npm.array(input_data)
		return input_np
	trainingData=readFile('../netId_project_2/datasets/Q3_data.txt')
	heig, weig,gen=[],[],[]
	# separating each parameter so can be used diffeently
	for i in trainingData:
		heig.append(float(i[0]))
		weig.append(float(i[1]))
		gen.append(i[3])

	def sigmoid(theta, inp):
		x = (1/(1+(mt.exp(-npm.dot(theta, inp)))))
		return x

	def const(alpha,trainData,gen):
		r,count= 0,0
		theta = npm.zeros(trainData.shape[1])
		for j in range(30):
			for i in range(len(trainData)):
				#taking probabilty by converting it to sigmoid function
				ra = sigmoid(theta, trainData[i])
				an = 1 if gen[i] == 'W' else 0
				if ra >= 0.5:
					if an == 1:
						r += 1
				if ra<=0.5:
					if an==0:
						r += 1
				# 
				trm1=(an-ra)
				theta += alpha * trm1 * trainData[i]
		return theta
	# learning rate
	alpha= 0.01
	#trainingData dataset
	# convert the list to numoy array
	X = npm.asarray([heig, weig]).T
	Y = npm.asarray(gen)
	# finding the constants
	st= const(alpha,X,Y)

	# making array to save the prediction for each given points.
	h,w,cl=[],[],[]
	for i in range(len(heig)):
		#converting the values to numberic and keeping thrashhold value for 0.5
		ans=sigmoid(st,[heig[i],weig[i]])
		#If the value is more than 0.5 than make it as male and make its value to be 1
		h.append(heig[i])
		w.append(weig[i])
		if ans>=0.5:
			cl.append('M')
		#If the value is less than 0.5 than make it as male and make its value to be 0
		else:
			cl.append('F')
	pt.figure(figsize=(5,9), dpi=100)
	ax = pt.axes(projection ='3d')
	for j in range(len(h)):
		if cl[j]=='M':
			ax.plot(h[j],w[j],'.g')
		else:
			ax.plot(h[j],w[j],'.r')
	pt.show()



	print('END Q3_D\n')


if __name__ == "__main__":
    main()
    