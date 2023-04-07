import numpy as np
import math
from matplotlib import pyplot as plt

def main():
	print('START Q2_D\n')
#Cleaning the dataset and seperating the column values.  	
	def	clean_data(line):
		return line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')
#fetching the file as a readme mode and cleaning the data
	def fetch_data(filename):
		with open(filename, 'r') as f:
			input_data = f.readlines()
			clean_input = list(map(clean_data, input_data))
			f.close()
		return clean_input
#reading the file.	
	def readFile(dataset_path):
		input_data = fetch_data(dataset_path)
		input_np = np.array(input_data)
		return input_np


#now we are converting the data in to the metrix
#cont is the fuction theta that is used to find the value of the constant functions. It is used to make a computerised Equation for theta .  
	def getConst(th_y,th_x,pt):
		# print(th_y)
		wtX,wt,temp2,temp1=[],[],[],[] #intialize the empty array
		for j in range(len(th_y)):
			tmp1=math.pow((th_x[j]-pt),2)
			tmp2=math.pow(0.204,2)
			tmp3=tmp2*2
			ans=np.exp(-tmp1/tmp3)
			wt.append(ans)
			# print(ans)
		for k in range(len(th_y)):
			temp2.append([th_y[k]*wt[k]])
			# print(wt[k])
			tmp=th_x[k]*wt[k]
			wtX.append(tmp)
			temp1.append([1,tmp])
		temp1=np.matrix(temp1)
		temp2=np.mat(temp2)
		invTemp1=np.linalg.pinv(temp1)
		answer=np.dot(invTemp1,temp2)
		# print(answer)
		a=np.array(answer)
		const=[]
		# print(a)
		for i in a:
			const.append(i[0])
		# print(wt)
		# print(wtX)
		# print(temp2)

		return const

#solution is a prediction function it is used to put the value of theta in to equation and mapping it with the x values in the array and 
#Used to finding the prediction values 
	def solution(const,th_x):
		ans=const[0]+const[1]*th_x
		return ans
	def wrng(gvn,fnd):
		d=0
		for x in range(len(gvn)):
			d+=(gvn[x]-fnd[x])**2
		wr=d/len(gvn)
		return wr
	# finding answers
	givenX,givenY=[],[] #intialize the empty array
	given=readFile('../netId_project_2/datasets/Q1_B_train.txt')
	count=0
	while count<20:
		count+=1
		givenX.append(float(given[count][0]))
		givenY.append(float(given[count][1]))
	findX,findY=[],[] #intialize the empty array
	find=readFile('../netId_project_2/datasets/Q1_C_test.txt')
	for i in range(len(find)):
		x=find[i][0] #taking value from Matrix
		y=find[i][1] #taking value from Matrix
		findX.append(float(x))
		findY.append(float(y))
	# print(findX)
	# print(findY)
	lsWr=10000
	x=0
	const=[]
	while x<(20):
		const.append(getConst(givenY,givenX, givenX[x]))
		x+=1
	# print(const)
	solved=[] #intialize the empty array
	x=0
	while x<(len(const)):
		prdicted=solution(const[x], givenX[x])
		solved.append(prdicted)
		x+=1
	wr=wrng(findY,solved)
	plt.scatter(givenX,solved,marker='.',label= 'solved')
	plt.scatter(givenX,givenY,marker='.',label= 'given')
	plt.show()

	# part B
	x=0
	const2=[] #intialize the empty array
	while x<(len(findX)):
		const2.append(getConst(findY,findX, findX[x]))
		x+=1
	solved2=[] #intialize the empty array
	solvedans=[] #intialize the empty array
	z=0
	while z<(len(const2)):
		prdicted=solution(const2[z], findX[z])
		solved2.append(prdicted)
		z+=1
	# wr=wrng(findY,solved2)

	plt.scatter(findX,solved2,marker='.',label= 'solved')
	plt.scatter(findX,findY,marker='.',label= 'given')
	plt.show()
	print('the error:',wr)


	print('END Q2_D\n')


if __name__ == "__main__":
    main()
    