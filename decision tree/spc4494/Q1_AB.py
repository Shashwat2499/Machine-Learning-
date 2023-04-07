import numpy as np
import math
traindatafile_1 = "./datasets/Q1_train.txt"  # train data path
testdatafile_1 = "./datasets/Q1_test.txt"  # test data path
#cleaning the dataset
def cleanLine(line: str):
    cleanLines = line.replace('(', '').replace(')', '').replace('\n', '').replace(' ','')  # remove undesired part from read input
    return cleanLines.split(",")
#reading the file
def readDataFromTxt(filePath: str):
    with open(filePath, 'r') as f:  # read data from path
        li = f.readlines()
        m = map(cleanLine, li)
        l = list(m)
        return np.array(l)  # return as np.array
# this function will read the data and predict labels
# here we are converting the features to values.
def getdata(filePath):
    data = readDataFromTxt(filePath)
    rft, getting = np.split(data, [-1], axis=1)
    # here we are predicting that is the value is 0 the W will be labeled otherwise M will be labeled.
    Predicting = getting == "M"
    getting = Predicting
    # we are returning the Answers of the prd_value .
    return rft.astype(float), getting.astype(float)
#declare the Decision tree class to find the current node , Right node , Left node and Best node .
class DT:
    def __init__(self,X=None,Y=None,nottrue1=None,ttrue1=None,dpt=1,maxDpt=math.inf,nType=0):
        self.nType = nType  #about current node
        self.maxDpt = maxDpt  # Max depth Allowed per tree
        self.dpt = dpt #maximum depth
        self.ttrue1 = ttrue1 #true Terms used
        self.rightChild = None  # remember the right node
        self.indexing_5 = None  # remember the best node
        self.nottrue1 = nottrue1 #wrong terms used
        if nType == 0: #for loop initialized and Give the ans zero
            self.nottrue1, self.ttrue1 = self.fc_1(X, Y)
        self.lft_chl = None  # remember the left node
        self.prd_value = None  # Predictions
        self.thr_val = None  # remember the best threshold value
        self.Infr_gain  = None  # remember the current node

    @staticmethod #here we have used the static method to call other class functions.
    def fc_1(X, Y):
        #classify in zero or one accordingn to true or false will be decided from Y
        wrongTermFilter , trueTermFilter = [] , []
        for ele in np.nditer(Y):
            if ele == 0:
                wrongTermFilter.append(True)
            else:
                wrongTermFilter.append(False)
            if ele == 1:
                trueTermFilter.append(True)
            else:
                trueTermFilter.append(False)
        nottrue1 = X[wrongTermFilter] #wrong terms classified
        ttrue1 = X[trueTermFilter] #return true terms that classified
        return nottrue1, ttrue1 #return both

    @staticmethod
   #this function is for calculating the entropy
    def computeEntropy(x, y):# calculating entropy for 2 values
        if x == 0 or y == 0: # calculating entropy for 2 values
            return 0 #return 0
        # Total number of features (ttrue1 + nottrue1)
        m = x + y
        p = x / m
        q = y / m
        etr = -(p) * math.log(p, 2) - (q) * math.log(q, 2) #computerised formula for returning the entropy
        return etr #final Entropy is returned
    @staticmethod
    def cmtfrf(ttrue1, nottrue1): #This is used for Calculating the entropy
        x = len(ttrue1) #it is for taking the features of other class
        y = len(nottrue1)
        return __class__.computeEntropy(x, y) #return the entropy
    def getSelfEntropy(self): #now we are finding the entropy of the node which is near
        return self.cmtfrf(self.ttrue1,self.nottrue1)
    @staticmethod
    def findMovingAvgOf2Ele(arr): #Using this function we are finding the threshold value
              return np.convolve(arr, [1, 1]) / 2
    @staticmethod
    def iG(be, lce, rce, lCFeaCount, rCFeaCount): #Using this function we are finding the information gain
        tf_count  = (lCFeaCount + rCFeaCount)
        l = lCFeaCount / tf_count #left child feature count and right child feature count
        r = rCFeaCount / tf_count
        i = be - (l * lce + r * rce)
        return i
    @staticmethod
    def infgain_find(bsc_ety1,tfeas_1,ffeas_1,tfeas_2,ffeas_2):
        lfce = __class__.cmtfrf(tfeas_1, ffeas_1) #finds left one
        rfce = __class__.cmtfrf(tfeas_2, ffeas_2) #finds right one
        blchildc = len(tfeas_1) + len(ffeas_1)
        brchildc = len(tfeas_2) + len(ffeas_2)
        return __class__.iG(bsc_ety1,lfce,rfce,blchildc,brchildc) # Calculating the information gain
    def Find_pred(self): #this functions checks that the leafnode is last node or not
        if len(self.ttrue1) >= len(self.nottrue1): #this functions checks that the leafnode is last node or not
            return True
        else:
            return False
    def processing (self): #now we are processing and training the data recursively
        a = (self.dpt > self.maxDpt)
        lwt= len(self.ttrue1)
        ltr= len(self.ttrue1)
        if  a or (lwt== 0 or ltr== 0):
            self.nType = 100 #initialize the leaf node 100
            self.prd_value = self.Find_pred()  # discontinue recursion
            return
        bsc_ety1 = self.getSelfEntropy() #now one more time calculate the entropy
        Col_full = self.ttrue1.shape[1]
        mxgain = -math.inf
        indexing_5 = 0
        bsthr_value  = ltsprts_less = ftsprts_less = ttsprts_mless =  ttsprts_lless =None
        qtr_ind = 0 #now we are giving 0 value to index
        while qtr_ind != Col_full:
            tffss_1 = self.ttrue1[::, qtr_ind] #we are trying to get from threshold values
            fvalye1 = self.nottrue1[::, qtr_ind] # maximizing our information gain
            colAllVals = np.union1d(tffss_1, fvalye1) #getting 2 lists of values and doing union among them
            psybt_1 = self.findMovingAvgOf2Ele(colAllVals) #forming a list of threshold values
            for tcrvalue in psybt_1: #loop for checking values one by one for threshold
                #we arer dividing values in to two parts like threshold
                tfeas_1 = self.ttrue1[tffss_1 < tcrvalue] #left < threshold values gives us left child
                ffeas_1 = self.nottrue1[fvalye1 < tcrvalue] #left < threshold values gives us left child
                tfeas_2 = self.ttrue1[tffss_1 >= tcrvalue]#left > threshold values gives us right child
                ffeas_2 = self.nottrue1[fvalye1 >= tcrvalue]#left > threshold values gives us right child
                Infr_gain  = self.infgain_find(bsc_ety1,tfeas_1,ffeas_1,tfeas_2,ffeas_2) #calculating the entropy
                #information gain.
                g1 = Infr_gain  < mxgain
                if not g1: #checking if we have same information gain or not
                    mxgain = Infr_gain #we are storing the values
                    indexing_5 = qtr_ind #we are storing the values
                    bsthr_value  = tcrvalue #we are storing the values
                    ltsprts_less = tfeas_1 #we are storing the values
                    ftsprts_less = ffeas_1 #we are storing the values
                    ttsprts_mless = tfeas_2 #we are storing the values
                    ttsprts_lless = ffeas_2 #we are storing the values
            qtr_ind = qtr_ind + 1 #incrementing by one
        self.thr_val = bsthr_value #getting the threshold value 
        self.indexing_5 = indexing_5 #getting the Index value 
        self.Infr_gain  = mxgain #getting the Information gain value 

        # Now we need to create left and right tree child and train them on divided data
        self.lft_chl = DT(nottrue1=ftsprts_less,ttrue1=ltsprts_less,dpt=self.dpt + 1,maxDpt=self.maxDpt,nType=(-1)) #checking the left child 
        self.rightChild = DT(nottrue1=ttsprts_lless,ttrue1=ttsprts_mless,dpt=self.dpt + 1,maxDpt=self.maxDpt,nType= 1) #checking the right child 
        self.lft_chl.processing () #process the left child 
        self.rightChild.processing () #process the right child 
        return self
    def Fin_predictions(self, tf): #making the predictions 
        if self.nType == 100 :
            return self.prd_value
        val = tf[self.indexing_5] # decission step to move left side or right
        bool = val > self.thr_val
        if bool:
            return self.rightChild.Fin_predictions(tf)
        else:
            return self.lft_chl.Fin_predictions(tf)
# In[ ]:
def myprediction(predicted_values, Final_values): #preducting the results according to the data
    predict_finalize = []     # firstly we initialize the empty array .
    for pred_fit in Final_values: # we are predicting the answers from this.
        perd_ans = predicted_values.Fin_predictions(pred_fit) # we are predicting the answers from this.
        predict_finalize.append(perd_ans) # we are predicting the answers from this.
        answer_of_prediction = predict_finalize # we are predicting the answers from this.
    return answer_of_prediction  # return the predictions
def finding_Accuracy(a_respon, predict_finalize): # Now we will find the accuracy using myprediction and Final Values
    total = len(a_respon)  # calculate the total answers
    predict_point = 0  # initialize the predict pointer to zero
    # we have to predict the features os the code
    for i in range(total):  # take a for loop
        if a_respon[i] == predict_finalize[i]:  # Predict the responses
            predict_point = predict_point + 1  # increment the prd_value point
    return predict_point / total # now we are returning the answers and getting the predictions
def main():
    print('START Q1_AB\n')
    '''
    Start writing your code here
    '''
    # fetching the dataset and using them
    training_data1, training_answers1 = getdata(traindatafile_1)  # training the model using training dataset
    Final_values, testing_answers1 = getdata(testdatafile_1)  # testing the model using testing dataset
    for maxDepth in range(1, 6):  # we are taking the for loop for range 1 to 6 so that we can get max depth
        print(f"DEPTH = {maxDepth}")  # printing the depth
        # Initialize model
        model = DT(X=training_data1,Y=training_answers1,maxDpt=maxDepth) #training the model
        model.processing ()  # traing the model from the dataset
        mak_pred_data_train = myprediction(model, training_data1)  # finding the accuracy using my predictions
        mak_pred_data_test = myprediction(model, Final_values)  # finding the accuracy using my predictions
        traindata_Acc = finding_Accuracy(training_answers1,mak_pred_data_train)  # finding the accuracy on train data using my predictions
        testdata_acc = finding_Accuracy(testing_answers1,mak_pred_data_test)  # finding the accuracy on test data using my predictions
        print(f"Accuracy | Train = {traindata_Acc:.3f} | Test =  {testdata_acc:.3f}")  # printing the functionn


# output  :
# _______________________________________
# # DEPTH = 1
# # Accuracy | Train = 0.840 | Test =  0.586
# # DEPTH = 2
# # Accuracy | Train = 0.840 | Test =  0.586
# # DEPTH = 3
# # Accuracy | Train = 0.860 | Test =  0.514
# # DEPTH = 4
# # Accuracy | Train = 0.920 | Test =  0.600
# # DEPTH = 5
# # Accuracy | Train = 1.000 | Test =  0.600
# _______________________________________
# Analysis    :
if __name__ == "__main__":
    main()