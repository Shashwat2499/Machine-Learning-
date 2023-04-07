import numpy as npm
import math as mt
import matplotlib.pyplot as pt

def main():
	print('START Q3_C\n')
	
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
	heig, weig, age,gen=[],[],[],[]
	# separating each parameter so can be used diffeently
	for i in trainingData:
		heig.append(float(i[0]))
		weig.append(float(i[1]))
		age.append(float(i[2]))
		gen.append(i[3])

	def sigmoid(theta, inp):
		# print('inp',inp)
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

	# making array to save the prediction for each given points.
	es=[]
	for i in range(len(gen)):
		rec=[]
		g=[]
		for j in range(len(gen)):
			if i!=j:
			#converting the values to numberic and keeping thrashhold value for 0.5
				rec.append([heig[j],weig[j],age[j]])
				g.append(gen[j])
		X = npm.asarray(rec)
		# print(X)
		Y = npm.asarray(g)
		st=const(alpha,X,Y)
		cl=[]
		for k in range(len(X)):
			ans=sigmoid(st,X[k])
			#If the value is more than 0.5 than make it as male and make its value to be 1
			if ans>=0.5:
				cl.append('M')
		#If the value is less than 0.5 than make it as male and make its value to be 0
			else:
				cl.append('F')
		e=0
		for z in range(len(cl)):
			if cl[z]!=Y[z]:
				e+=1
		es.append([st.tolist(),e/len(cl)])
		minE=mt.inf
		req=[]
		for j in range(len(es)):
			if minE>es[j][1]:
				minE=es[j][1]
				req=es[j]
		requiredPredictedPts=[]
	arr=[]
	for x in range(len(heig)):
		arr.append([heig[x],weig[x],age[x]])
	reqArr=npm.asarray(arr)
	print('minimum error:',minE)
	# print('reqA:',reqArr)
	for x in range(len(reqArr)):
		tmps=sigmoid(req[0],reqArr[x])
		if tmps>=0.5:
			ans=1
		else:
			ans=0
		# print(tmps)
		requiredPredictedPts.append([reqArr[x].tolist(),ans])
	ax = pt.axes(projection ='3d')
	print(es)
	for z in range(len(requiredPredictedPts)):
		# print('reqpredic',requiredPredictedPts)
		if requiredPredictedPts[z][1]==1:
			ax.plot(requiredPredictedPts[z][0][0],requiredPredictedPts[z][0][1],requiredPredictedPts[z][0][2],'.b')
		else:
			ax.plot(requiredPredictedPts[z][0][0],requiredPredictedPts[z][0][1],requiredPredictedPts[z][0][2],'.y')
	pt.show()
	
	print('END Q3_C\n')


if __name__ == "__main__":
    main()
    