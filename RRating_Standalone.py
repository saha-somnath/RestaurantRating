__author__ = 'somnath'


'''
MIT License

Copyright (c) 2020 Somnath Saha

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''


import numpy as np
import cv2
import sys
import os
import glob
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
#from sklearn.neural_network import MLPClassifier
import time

'''
Program Name: RRating_Standalone.py
Description : This program does following major thins for Restaurant rating Project
              - Classification;
                  - Feature Extraction from each image
                  - Train the classifier with Training Image
                  - Test the uploaded image by predicting its class.

              - Choosing Suitable Parameter for classifier:
                - Perform cross validation once to evaluate classifier and system.
                - Classifier: SVM 
                  - Parameter { gamma = 0.5 , C=1.0, kernel ='rbf' }

              - Testing: Train the uploaded image, predict the class and find rating.
'''



path = "images"
testImagePath = "test_images"

# Feature length limit
featureVectorSize = 10000

# Restaurant Label - Each one having unique label
restaurantLabel = {
    'Starbucks' : 0,
    'PaneraBread' : 1,
    'DunkinDonuts' : 2,
    'Chipotle' : 3,
    'BonefishGrill' : 4,
    'PizzaHut': 5
}

# Number of Restaurant
restaurantNumber = len(restaurantLabel)


#####################################################################################
'''
Function Name: featureExtraction()
Input Args   : <Sports Action Path>, <Action name>, <Training/ Validation>,
Returns      : <Array: Feature List>
Description  : This function extract features from each restaurant image and consolidated them.
               While it extract features, it add label to feature at the beginning of feature vector based on Restaurant
               Label. It helps to keep tack of feature and corresponding label while shuffle the features during
               cross validation.

               - I have used histogram of oriented gradient (HOG) method to extract the features.
                 Following methods from cv2 have been used.
                 hogDescriptor = cv2.HOGDescriptor()
                  - It takes default parameter values as Window Size= 64 x 128, block size= 16x16,
                    block stride= 8x8, cell size= 8x8, bins= 9
                 hist = hogDescriptor.compute(gray)
                  - Returns the list of histogram

               - Sorted the Histogram and taken top 10000 for evaluation.
'''

def getFeatures(restaurantName, image,  type):

    # Read Frame
    frame = cv2.imread(image)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #hog = cv2.HOGDescriptor((32,64), (16,16), (8,8), (8,8), 9)
    hog = cv2.HOGDescriptor()
    hogHist = hog.compute(gray)

    sortedHogHist = np.sort(hogHist, axis=None)
    keyFeatures = sortedHogHist[- featureVectorSize : ]

    if type == "Trng":
        keyFeatures = np.insert(keyFeatures, 0, restaurantLabel[restaurantName])

    return    keyFeatures

'''
Function Name: getListOfDir()
Input Args   : < Directory Path >
Return       : <Array: List of Directory >
Description  : This function returns all the directories under the specified paths
'''
def getListOfDir(path):
    # Read each Restaurants directory
    dirs  = os.listdir(path)

    filtered_dir  = []
    # Remove . .. and hidden directory
    for dir in dirs:
        if not dir.startswith("."):
            filtered_dir.append(dir)

    return filtered_dir

'''
Function Name: getRestaurantName()
Input Args   : < Restaurant Index>
Return       : < Restaurant Name>
Description  : This function returns the name of the Restaurant based on index value
'''
def getRestaurantName(rIndex):

    keys   = restaurantLabel.keys()
    for key in keys:
        if rIndex == restaurantLabel[key]:
            return key


'''
Function Name: evaluation()
Input Args   : < 1D Array: Truth>, <1D Array: Predicted>, < Restaurant Label Index>
Return       : <Accuracy>,<Precision>,<Recall>
Description  :  This function calculate evaluation metrics accuracy, precision and recall based on
                True Positive (TP), False Positive (FP), False Negative (FN) and True Negative (TN) rate.

               Precision   =  TP / ( TP + FP )
               Recall      =  TP / ( TP + FN )
               Accuracy    = ( TP + TN ) / ( TP + FN + FP + TN )
'''
def evaluation( truth, predicted, categoryIndex ):

    TP = 1 # True Positive
    FP = 1 # False Positive
    FN = 1 # False Negative
    TN = 1 # True Negative

    # Categories are Rest1=>0, Rest2=> 1, Rest3=>2  etc..
    for fIndex in range(len(truth)):

        if ( int(predicted[fIndex]) == categoryIndex):
            # TP=> when P[i] = T[i] = Ci
            if (int(truth[fIndex]) == int (predicted[fIndex])):
                TP += 1
            else:
                FP += 1
        else:
            if ( int ( truth[fIndex]) == categoryIndex ):
                FN += 1
            else:
                TN += 1
    #print "\t\t TP-{0}, FP-{1} , FN-{2}, TN-{3}".format(TP,FP,FN,TN)

    if ( TP > 1):
        TP -=1
    if ( FP > 1):
        FP -=1
    if ( FN > 1):
        FN -=1
    if ( TN > 1):
        TN -=1

    # Calculate precision - what fraction of prediction is correct
    precision = TP / float ( TP + FP )
    # Calculate recall
    recall    = TP / float ( TP + FN )
    #Calculate Accuracy
    accuracy =  ( TP + TN ) / float ( TP + FP + FN + TN )

    print "\t Accuracy :", accuracy
    print "\t Precision:", precision
    print "\t Recall   :", recall

    return accuracy, precision, recall

'''
Function Name: crossValidation()
Input Args   : < Array: Feature and Label List - Fits element of vector indicates action label and rest are for features>
Retrun       : None
Description  : It perform Leave-One-Out ( LOO ) cross validation.
               First, I shuffle the feature list which contains features as well as corresponding label at the very
               first element of the feature vector . The complete set of shuffled features are divided equally
               into k=6 sub parts. k-1 subset is used for training and one subset is used for validation. I iterate the
               process for k=6 times with different subset combinations for training and validation.

               Evaluation Metrics:
                At each iteration, evaluation metrics sensitivity, specificity and accuracy are calculated
                based on True Positive (TP), False Positive (FP), False Negative (FN) and True Negative (TN) rates.

               Precision   =  TP / ( TP + FP )
               Recall      =  TP / ( TP + FN )
               Accuracy    = ( TP + TN ) / ( TP + FN + FP + TN )

               At the end of all iterations of cross validation, I average them all to get the average rate.
'''
def crossValidation( featureAndLabelList):

    # Randomize the sample
    np.random.shuffle(featureAndLabelList)


    # Evaluation Metrics
    accuracy    = 0.0
    precision   = 0.0
    recall      = 0.0


    # split feature set in equal subsets same a number of categories for cross validation
    subsetLength =  len(featureAndLabelList) / restaurantNumber

    for rIndex in range(len(restaurantLabel)):

        print "INFO: Cross Validation Iteration - ", rIndex
        trainigSet = []
        valdationSet = []
        feature = []
        label   = []


        if ( rIndex == 0 ):
            trainigSet = featureAndLabelList[1*subsetLength:]
            valdationSet = featureAndLabelList[0: subsetLength]
        elif ( rIndex == ( restaurantNumber -1 )):
            trainigSet = featureAndLabelList[:( restaurantNumber -1 )*subsetLength]
            valdationSet = featureAndLabelList[( restaurantNumber -1 )*subsetLength : ]
        else:
            trainigSet = np.concatenate ((featureAndLabelList[:rIndex * subsetLength] , featureAndLabelList[(rIndex + 1) * subsetLength: ]), axis=0 )
            valdationSet = featureAndLabelList[rIndex * subsetLength : (rIndex + 1 ) * subsetLength]

        # Get all features in a array
        for featureAndLabel in trainigSet:
            label.append(int(featureAndLabel[0]))
            feature.append(np.delete(featureAndLabel, 0))


        # Train model
        print "INFO: Training ... "
        clf = svm.SVC(gamma=0.5, C=1.0)
        #clf=RandomForestClassifier(n_estimators=6)
        #clf = AdaBoostClassifier(n_estimators=100)
        clf.fit(feature,label)


        # Prepare validation feature and label to be predicted
        print "INFO: Prediction for ", getRestaurantName(rIndex)
        vFeatureList = []
        vLabelList   = [] # Ground Truth
        for featureAndLabel in valdationSet:
            vFeatureList.append(featureAndLabel[1:].tolist())
            vLabelList.append(featureAndLabel[0])

        predictedLabel = clf.predict(vFeatureList)

        # predict validation set and calculate accuracy
        print "\t **** Evaluation **** "
        # Evaluation < Truth>, <Predicted>, <Restaurant Index>
        ( a , p , r ) = evaluation(vLabelList , predictedLabel.tolist() , rIndex)

        accuracy  += a
        precision += p
        recall    += r

    # Average evaluation metrics
    avgAccuracy = accuracy / len(restaurantLabel)
    avgPrecision = precision / len(restaurantLabel)
    avgRecall    = recall / len(restaurantLabel)

    print "\t*** Evaluation ***"
    print "\t Average Accuracy : ", avgAccuracy
    print "\t Average Precision: ", avgPrecision
    print "\t Average Recall   : ", avgRecall



'''
Function Name: getImageList()
Input Args   : <Image Directory>
Retrun       : List of Images
'''
def getImageList(imageDirectory):

    # Find different type of images
    rImages = glob.glob(imageDirectory + "/*.jpg")
    rImages +=  glob.glob(imageDirectory + "/*.jpeg")
    rImages +=  glob.glob(imageDirectory + "/*.png")

    return rImages



def main():

    # Read all the images from Image directory
    restaurantImageDirs = getListOfDir(path)
    print "INFO: Restaurants:", restaurantImageDirs

    fxTime = time.time()
    # Key Feature Lists for all the restaurant images
    keyFeatureList = []

    for rImageDirName in restaurantImageDirs:

        print "INFO: Extracting Key Features for restaurant: {0}".format(rImageDirName)
        rImagePath = path + "/" + rImageDirName

        # Read all images from a restaurant and create a feature + label mapped data structure
        rImages = getImageList(rImagePath)
        for image in rImages:

            featureVector = getFeatures(rImageDirName,image, 'Trng')

            if len(featureVector) == (featureVectorSize + 1):
                # Add all the features from feature vector to a common list
                keyFeatureList.append(featureVector)

    print "Total Features:", len(keyFeatureList)
    print "Feature Exraction Time: %s Seconds" % ( time.time() - fxTime )

    crossValidationStartTime = time.time()
    # Perform Cross validation Only once to check parameter
    print "\t **** Cross Validation **** "
    crossValidation(keyFeatureList)
    print "INFO: Cross validation Time: %s seconds" % ( time.time() - crossValidationStartTime )


    # Train the classifier
    clf = svm.SVC(gamma=0.5, C=1.0)
    #model=RandomForestClassifier(n_estimators=6)
    #model = AdaBoostClassifier(n_estimators=100)

    trngFeature = []
    trngLabel = []

    # Shuffle Data
    np.random.shuffle(keyFeatureList)
    # Separate feature and label from each vector and store in new feature & label arrays.
    for featureAndLabel in keyFeatureList:
        #print "Y:", int(featureAndLabel[0])
        trngLabel.append(int(featureAndLabel[0]))
        trngFeature.append((np.delete(featureAndLabel, 0)).tolist())

    print "INFO: Train the model", len(trngFeature)
    trainStartTime = time.time()
    clf.fit(trngFeature, trngLabel)
    print "INFO: Training Time: %s seconds" % ( time.time() - trainStartTime )

    ## This Part is only for Initial Testing. ##
    # Test multiple images ( For Testing Only )
    print "\t **** Testing ***** "
    testPath = "/Users/somnath/MY_PROG/ML/FinalProject/Test"

    testImageDirs = getListOfDir(testPath)

    correctPrediction = 0
    imageCount        = 0

    for testImageDirName in testImageDirs:

        testImagePath = testPath + "/" + testImageDirName
        print "INFO: Predicting images - ", testImageDirName
        # Read all images from a restaurant and create a feature + label mapped data structure
        images = getImageList(testImagePath)
        for image in images:
            testFeatureVector = getFeatures(testImageDirName,image, 'Test')
            if len(testFeatureVector) == featureVectorSize :
                predictedLabel = clf.predict([testFeatureVector])

                result = ""
                if predictedLabel[0] == restaurantLabel[testImageDirName]:
                    result = "CORRECT"
                    correctPrediction += 1
                else:
                    result = "WRONG"

                imageCount += 1
                print "\t Predicted Restaurant:{0}-> {1} -> {2} ->{3}".format((image.split('/'))[-1],predictedLabel,getRestaurantName(predictedLabel[0]), result )
            else:
                print "ERROR: Feature Length ", len(testFeatureVector)

    print "Total Sample: {0} , Correct: {1} , Wrong: {2}".format(imageCount, correctPrediction, (imageCount - correctPrediction))







if __name__ =="__main__":
    main()
