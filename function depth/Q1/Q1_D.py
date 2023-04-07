import math
from matplotlib import pyplot as plt
import numpy as np


def main():
	print('START Q1_D\n')
#Cleaning the dataset and seperating the column values.	
	def clean_data(line):
		return line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')
# fetching the data for the file
	def fetch_data(filename):
		with open(filename, 'r') as f:
			input_data = f.readlines()
			clean_input = list(map(clean_data, input_data))
			f.close()
		return clean_input
#reading the file using numpy 
	def readFile(dataset_path):
		input_data = fetch_data(dataset_path)
		input_np = np.array(input_data)
		return input_np

#we are converting the data in to the metrix
#cont is the fuction theta that is used to find the value of the constant functions.
# It is used to make a computerised Equation for theta .  
	def getConst(th_x,th_y,inc,fre):
		temp2=[] #initializa the empty array
		bcm=[] #initializa the empty array
		for i in th_y:
			bcm.append([i]) #added the element in to the array
		bcm=np.mat(bcm)   #used to interpret a given input as a matrix.
		for j in range(len(th_y)):
			temp=[] #initializa the empty array
			temp.append(1)
			for i in range(1,fre+1):
				ans=math.pow(math.sin(i*th_x[j]*inc),2)
				temp.append(ans) #added the element in to the array
			temp2.append(temp) #added the element in to the array
		temp2=np.mat(temp2)   #used to interpret a given input as a matrix.
		#function used for psuedo Inverse. 
		invtemp2=np.linalg.pinv(temp2) 
		const=np.dot(invtemp2,bcm) #cont is the fuction theta that is used to find the value of the constant functions.
		array=const.tolist()
		ans=[] #initializa the empty array
		for i in array:
			ans.append(i)
		return ans
	def solution(const,inp,k):
		answer=[] #initializa the empty array
		for j in range(len(inp)):
			ans=const[0][0]
			for i in range(1,len(const)):
				ans+=float(const[i][0])*float(math.pow((math.sin(k*i*inp[j])),2))
			answer.append(ans)
		array=np.array(answer) 
		list=array.tolist() #convert a given array to an ordinary list
		ans=[] #initializa the empty array
		for i in list:
			ans.append(i)
		return ans
		#finding the error functions 
	def wrng(gvn,fnd):
		d=0
		for x in range(len(gvn)):
			d+=math.sqrt(abs(gvn[x]-fnd[x]))
		wr=d/(len(gvn)*3)
		return wr

	# finding answers
	givenX,givenY=[],[] #initializa the empty array
	given=readFile('../netId_project_2/datasets/Q1_B_train.txt')
	count=0
	#taken an while condition to append the elements till it get false
	while count<21:
		count+=1
		givenX.append(float(given[count][0]))
		givenY.append(float(given[count][1]))
	findX,findY=[],[]  #initializa the empty array
	find=readFile('../netId_project_2/datasets/Q1_c_test.txt')
	#taken the for loop to append the elements 
	for i in range(len(find)):
		x=find[i][0]
		y=find[i][0]
		#giving the output
		findX.append(float(x))
		findY.append(float(y))
	lsWr=10000
	x=1
	#conditional statements and then calling the appropriate functions as per requirements.
	ans=[]
	prid=[]
	while x<11:
		y=0
		while y<7:
			#giving the parameters and calling it by function name 
			#cont is the fuction theta that is used to find the value of the constant functions
			const=getConst(givenY,givenX,x,y)
			solved=solution(const,givenX,x)
			tpWr=wrng(givenY,solved)
			#least error finding 
			ans.append([x,y,tpWr])
			prid.append(solved)
			lsWr=min(tpWr,lsWr)
			y+=1
		x+=1
		# finding the error function for the requirements 
	for i in ans:
		print(f'for k={i[0]}, d={i[1]}, error={i[2]}\n')
	print('least error was:-',lsWr)

	print('END Q1_D\n')


if __name__ == "__main__":
    main()