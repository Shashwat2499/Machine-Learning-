import numpy as np
import math
from matplotlib import pyplot as plt

def main():
	print('START Q2_AB\n')
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
		wtX,wt,temp2,temp1=[],[],[],[] #taken an empty array
		for j in range(len(th_y)): 
			tmp1=math.pow((th_x[j]-pt),2) #returns the power of the value 
			tmp2=math.pow(0.204,2) #returns the power of the value 
			tmp3=tmp2*2 
			ans=np.exp(-tmp1/tmp3) # taking E raise to power definition.
			wt.append(ans)
			# print(ans)
			#loop fot appendind the answers to the array matrix
		for k in range(len(wt)):
			tmp=th_x[k]*wt[k] #using the theta value in to the equation
			wtX.append(tmp) #added the element in to the array
			temp1.append([1,tmp]) #added the element in to the array
			temp2.append([th_y[k]*wt[k]])
		temp1=np.matrix(temp1) 
		temp2=np.mat(temp2)
		#used for psuedo inverse 
		invTemp1=np.linalg.pinv(temp1)
		answer=np.dot(invTemp1,temp2)
		a=np.array(answer)
		const=[] 
		for i in a:
			const.append(i[0])
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
	givenX,givenY=[],[]
	given=readFile('../netId_project_2/datasets/Q1_B_train.txt')
	for i in given:
		givenX.append(float(i[0]))
		givenY.append(float(i[1]))
	const=[]
	x=0
	while x<(len(givenX)):
		const.append(getConst(givenY,givenX, givenX[x]))
		x+=1
	# print(const)
	solved=[]
	x=0
	while x<(len(const)):
		print(const[x])
		prdicted=solution(const[x], givenX[x])
		solved.append(prdicted)
		x+=1
	wr=wrng(givenY,solved)
	# print(solved)
	# plt.scatter(givenX,solved,marker='.',label= 'predicted')
	# plt.scatter(givenX,givenY,marker='.',label= 'orignal')
	# plt.show()
	print('the error:',wr)

	print('END Q2_AB\n')


if __name__ == "__main__":
    main()
