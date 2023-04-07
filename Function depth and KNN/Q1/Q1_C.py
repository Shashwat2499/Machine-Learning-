import math
import numpy as np
import matplotlib.pyplot as plt

def main():
	print('START Q1_C\n')
	#Cleaning the dataset and seperating the column values.
	def clean_data(line):
		return line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')
	def wrng(gvn,fnd):
		d=0
		for x in range(len(gvn)):
			d=(gvn[x]-fnd[x])**2
		wr=5*d/len(gvn)
		return wr
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
		temp2=[]  #initializa the empty array 
		bcm=[]  #initializa the empty array
		for i in th_y:
			bcm.append([i]) #added the element in to the array
		bcm=np.mat(bcm)
		for j in range(len(th_y)):
			temp=[]  #initializa the empty array
			temp.append(1) #added the element in to the array
			for i in range(1,fre+1):
				ans=math.pow(math.sin(i*th_x[j]*inc),2)
				temp.append(ans) #added the element in to the array
			temp2.append(temp) #added the element in to the array
		temp2=np.mat(temp2)
		invtemp2=np.linalg.pinv(temp2)
		#cont is the fuction theta that is used to find the value of the constant functions.
		const=np.dot(invtemp2,bcm)
		array=const.tolist()
		ans=[]  #initializa the empty array
		for i in array:
			ans.append(i)
		return ans
	#as given in the requirements  1, sin2(x), sin2(k ∗x), sin2(2 ∗ k ∗ x),..., where k is effectively the frequency increment	
	def solution(const,inp,k):
		answer=[]  #initializa the empty array
		for j in range(len(inp)):
			ans=const[0][0]
			for i in range(1,len(const)):
				#computerised equation for counting the linear regression . 
				ans+=float(const[i][0])*float(math.pow((math.sin(k*i*inp[j])),2))
			answer.append(ans)
		array=np.array(answer)
		list=array.tolist()
		ans=[]
		for i in list:
			ans.append(i)
		return ans

	# finding answers
	givenX,givenY=[],[]
	given=readFile('../netId_project_2/datasets/Q1_B_train.txt')
	for i in given: #loop for appending the values in to the array 
		givenX.append(float(i[0])) 
		givenY.append(float(i[1]))
	findX,findY=[],[] #empty array intialization 
	find=readFile('../netId_project_2/datasets/Q1_c_test.txt')
	for i in range(len(find)): #transferring to array 
		x=find[i][0] 
		y=find[i][0]
		#giving the output
		findX.append(float(x)) #added the element in to the array
		findY.append(float(y)) #added the element in to the array
	lsWr=10000
	ans=[]
	# question part B
	x=1
	#conditional statements and then calling the appropriate functions as per requirements.
	while x<11:
		y=0
		while y<7:
			#cont is the fuction theta that is used to find the value of the constant functions
			const=getConst(givenY,givenX,x,y)
			solved=solution(const,givenX,x)
			tpWr=wrng(givenY,solved)
			#least error finding 
			lsWr=min(tpWr,lsWr)
			ans.append([x,y,tpWr])
			y+=1
		x+=1
	print(ans)
	#printing the least error
	print('least error was:-',lsWr)
	print('END Q1_C\n')

#calling the main function S
if __name__ == "__main__":
    main()
    