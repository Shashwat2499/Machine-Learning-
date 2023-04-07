from matplotlib import pyplot as plt
import numpy as np
import math

def main():
	print('START Q1_AB\n')
#Cleaning the dataset and seperating the column values.  
	def clean_data(line):
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
	def getConst(th_x,th_y,inc,fre):        
		temp2=[]  #taken an empty array
		bcm=[]    #taken an empty array
		for i in th_y: #loop
			bcm.append([i])    #added the element in to the array
		bcm=np.mat(bcm) #used to interpret a given input as a matrix.
		for j in range(len(th_y)): #loop
			temp=[] #taken an empty array
			temp.append(1) #added the element in to the array
			for i in range(1,fre+1): #loop
				ans=math.pow(math.sin(i*th_x[j]*inc),2) #an array containing elements of the first array raised to the power element of the second array.
				temp.append(ans) #added the answer in the array
			temp2.append(temp)  #added the temp variable to array
		temp2=np.mat(temp2)
		#now we are doing the psuedo inverse of the metrix
		invtemp2=np.linalg.pinv(temp2)
		#performing the dot product of two arrays
		const=np.dot(invtemp2,bcm) 
		array=const.tolist() #convert a given array to an ordinary list
		ans=[]
		for i in array:
			ans.append(i)
		return ans
	def solution(const,inp,k):
		answer=[]
		for j in range(len(inp)):
			ans=const[0][0]
			for i in range(1,len(const)):

				#as given in the requirements  1, sin2(x), sin2(k ∗x), sin2(2 ∗ k ∗ x),..., where k is effectively the frequency increment
				ans+=float(const[i][0])*float(math.pow((math.sin(k*i*inp[j])),2))
			answer.append(ans)
		array=np.array(answer)
		list=array.tolist()
		ans=[]
		for i in list:
			ans.append(i)
		return ans

	# finding answers 
	givenX,givenY=[],[] #intialize the empty array
	given=readFile('../netId_project_2/datasets/Q1_B_train.txt')
	#giving the output
	k=int(input('give frequency increment'))
	d=int(input('give the deepth'))
	for i in given:
		givenX.append(float(i[0]))
		givenY.append(float(i[1]))
#cont is the fuction theta that is used to find the value of the constant functions. It is used to make a computerised Equation for theta .
	const=getConst(givenY,givenX,k,d)
#---------------------------------------------------------------------------------------------------------------------------#
	# question part B
	x=1
	while x<11:
		y=0
		while y<7:
			const=getConst(givenY,givenX,x,y)
			solved=solution(const,givenX,x)
			plt.scatter(givenX,solved,marker='.',label= y)
			plt.legend()
			y+=1
		x+=1
		plt.show()
	print('END Q1_AB\n')


if __name__ == "__main__":
    main()