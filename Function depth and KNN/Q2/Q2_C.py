import numpy as np
import math
from matplotlib import pyplot as plt

def main():
	print('START Q2_C\n')
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
		wtX,wt,temp2,temp1=[],[],[],[] #taken an empty arrays
		for j in range(len(th_y)): 
			tmp1=math.pow((th_x[j]-pt),2) #array containing elements of the first array raised to the power element of the second array.
			tmp2=math.pow(0.204,2) #array containing elements of the first array raised to the power element of the second array.
			tmp3=tmp2*2
			ans=np.exp(-tmp1/tmp3) #exponential function 
			wt.append(ans)
			# print(ans)
		for k in range(len(th_y)):
			temp2.append([th_y[k]*wt[k]])
			# print(wt[k])
			tmp=th_x[k]*wt[k]
			wtX.append(tmp)
			temp1.append([1,tmp])
		temp1=np.matrix(temp1)
		temp2=np.mat(temp2) #used to interpret a given input as a matrix.
		#function used for psuedo Inverse
		#now we are doing the psuedo inverse of the metrix
		invTemp1=np.linalg.pinv(temp1)
		#performing the dot product of two arrays
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
	givenX,givenY=[],[] #intialized the empty array
	given=readFile('../netId_project_2/datasets/Q1_B_train.txt')
	count=0
	#giving the output
	while count<21:
		count+=1
		givenX.append(float(given[count][0]))
		givenY.append(float(given[count][1]))
	findX,findY=[],[] #intialized the empty array
	find=readFile('../netId_project_2/datasets/Q1_C_test.txt')
	for i in range(len(find)):
		x=find[i][0] #taking value from Matrix
		y=find[i][1]  #taking value from Matrix
		findX.append(float(x))
		findY.append(float(y))
	# print(findX)
	# print(findY)
	lsWr=10000
	x=0
	const=[] #intialized the empty array
	while x<(len(findX)):
		const.append(getConst(findY,findX, findX[x]))
		x+=1
	# print(const)
	solved=[] #intialized the empty array
	#now we are pedicting the label . 
	x=0
	while x<(len(const)):
		prdicted=solution(const[x], findX[x])
		solved.append(prdicted)
		x+=1
		#now we are pedicting the label .
	wr=wrng(findY,solved)
	plt.scatter(findX,solved,marker='.',label= 'predicted')
	plt.scatter(findX,findY,marker='.',label= 'orignal')
	plt.show()
	#printing the Error 
	print('the error:',wr)


	print('END Q2_C\n')


if __name__ == "__main__":
    main()
    