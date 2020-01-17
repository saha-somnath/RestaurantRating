import os
import numpy as np
import cv2
import sys
import os
import glob
from sklearn import svm
from scipy.stats import mode
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import AdaBoostClassifier
import time

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


from flask import Flask, request, redirect, url_for, jsonify, flash
from werkzeug.utils import secure_filename


'''
Program Name: RRating_Backend.py
Description : This is the backend serverside code for restaurant rating project.
              It perform following task.
              - Code Runs in Flask webserver
              - It provide file upload mechanism for both SmartPhone app and browser
              - Classify the test image
              - Find the rating of the restaurant 
              - Use YELP business dataset or API with secret business key.
'''


UPLOAD_FOLDER = 'src/Uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY']    = 'super secret key'
app.config['SESSION_TYPE']  = 'memcached'
#app.config['SESSION_TYPE']  = 'filesystem'
app.config['IS_TRAINED']    = 0
app.config['RESPONSE_TIME'] = 0.0



# Restaurant Label
restaurantLabel = {
    'Starbucks'     : 0,
    'PaneraBread'   : 1,
    'DunkinDonuts'  : 2,
    'Chipotle'      : 3,
    'BonefishGrill' : 4,
    'PizzaHut'      : 5
}

# SVM Params
clf = svm.SVC(gamma=0.5, C=1.0)

# Training Image Path
#path = "/Users/somnath/MY_PROG/ML/FinalProject/Image"
path = "images"

featureVectorSize = 10000


# Rating -Testing
restaurantRating =  {
    'Starbucks' : 4.5,
    'PaneraBread' : 4.2,
    'DunkinDonuts' : 3.8,
    'Chipotle' : 4.0,
    'BonefishGrill' : 4.1,
    'PizzaHut': 3.6
}

###############################################

'''
Function Name: allowed_file()
Input Args   : < Test Image>
Return       : < File Name >
Description  : Return the allowedd flie 

'''

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS



'''
Function Name: predictLabel()
Input Args   : < Test Image>
Return       : < Restaurant Name>
Description  : This function is for Web Interface for browser.
               - upload the restaurant image to the upload dir of webserver.
               - Process Image ( Feature Extraction , Class Prediction, finding Restaurant Name  )
               - Get rating from restaurant name.
               - Send the Restaurant Name and Rating as response to the app
'''

@app.route('/', methods=['GET', 'POST'])
def upload_file():

    # Create Header Part of HTML
    part1 = " <!doctype html> <title>Restaurant Rating</title>"
    part2 = ""
    part3 = '''<body>
                 <h5>Upload New File</h5>
                 <form action="" method=post enctype=multipart/form-data>
                   <input type=file name=file>
                   <input type=submit value=Upload>
                 </form>
               </body>
            '''
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            
            filePath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filePath)

            # Train The Model
            if not app.config['IS_TRAINED']:
                trainModel()
                app.config['IS_TRAINED'] =  1
            
            uploadedImage = os.path.join(app.config['UPLOAD_FOLDER'], filename )
            # ProcesoImage
            rName = processImage(uploadedImage)
            print "Restaurant Name  :", rName
            
            part2 = ""
            if  rName == 'None' :
                part2 =  "<h5> Invalid File </h5>"
            else:
                # Return Restaurant Rating
                rRating = getRating( rName )
                print "Restaurant Rating:", rRating

            
                part2 = "<h5> Restaurant Name: " +  rName + "</h5>" + "<h5>Rating:" + str(rRating) + " </h5>" \
                        + "<h5>Response Time:" + str(app.config['RESPONSE_TIME']) + " Seconds </h5>" 

            return part1 + part2 + part3
            

    return part1 + part3  
 

'''
Function Name: upload()
Input Args   : < Test Image>
Return       : < Restaurant Name>
Description  : This function is for SmartPhone apps.
               - Upload the file to the upload dir of webserver
               - Process Image ( Feature Extraction , Class Prediction, finding Restaurant Name  )
               - Get rating from restaurant name.
               - Send the Restaurant Name and Rating as response to the app
'''

@app.route('/upload', methods=['GET', 'POST'])
def upload():

         
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            print "ERROR: No File Part"
            return "ERROR: No File Part"
                       
        file = request.files['file']
        # if user does not select file, browser also. submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return "No selected file"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            uploadedImage = os.path.join(app.config['UPLOAD_FOLDER'])
            # Process Image
            rName = processImage(uploadedImage)
            # Return Restaurant Rating
            rRating = getRating( rName )
                
            # Display Message with Restaurant Name and Rating
            displayMsg = "Name:" + rName + "\nRating:" + str(rRating)
            return displayMag
    else:
        return "GET Method"
 

'''
Function Name: featureExtraction()
Input Args   : <Sports Action Path>, <Action name>, <Training/ Validation>,
Returns      : <Array: Feature List>
Description  : This function extract features from each frames of a video and consolidated them.
               While it extract features, it add label to feature at the beginning of feature vector based on Sports
               Action Type. It helps to keep tack of feature and corresponding label while shuffle the features during
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

    #print "HOG 2D:", hogDescriptor
    #sortedHogDescriptor = sorted(hogDescriptor)
    sortedHogHist = np.sort(hogHist, axis=None)

    keyFeatures = sortedHogHist[- featureVectorSize : ]

    #print keyFeatures

    if type == "Trng":
        keyFeatures = np.insert(keyFeatures, 0, restaurantLabel[restaurantName])

    return    keyFeatures


'''
Function Name: predictLabel()
Input Args   : < Test Image>
Return       : < Restaurant Name>
Description  : This function predict the label for a restaurant image and returns the name of the  Restaurant 

'''
def processImage(uploadedImage):
    
    # Extract The feature
    print "\t **** Testing ***** "
    print "Test Image:", uploadedImage
    
    
    predictionStart =  time.time()

    testImageFeature = getFeatures("Unknown",uploadedImage, 'Test')
    
    if len( testImageFeature) != featureVectorSize:
        return "None"

    # Predict the class of Image
    predictedLabel = clf.predict([testImageFeature])
    restaurantName = getRestaurantName(predictedLabel[0])

    predictionTime =  ( time.time() - predictionStart )
    print "INFO: Prediction Time : %s seconds "  % predictionTime
    app.config['RESPONSE_TIME']  = predictionTime

    return restaurantName


'''
Function Name: predictLabel()
Input Args   : < Test Image>
Return       : < Restaurant Name>
Description  : This function predict the label for a restaurant image and returns the name of the  Restaurant 

'''
def getRating(restaurantName):

    # Call YELP for rating
    
    
    # Retrun rating
    return restaurantRating[restaurantName]


'''
Function Name: trainModel()
Input Args   : < None>
Return       : < None>
Description  : This function train the classifier ( SVM ) with features and corresponding label.
               - Features gets extracted from images to be trained.
               - CLassifier gets trained with features and corresponding label
              
'''
def trainModel():

    fxStart = time.time()

    # Read all the images from Image directory
    restaurantImageDirs = getListOfDir(path)
    print "INFO: Restaurants:", restaurantImageDirs

    # Key Feature Lists for all the restaurant images
    keyFeatureList = []

    for rImageDirName in restaurantImageDirs:

        print "INFO: Extracting Key Features for restaurant: {0}".format(rImageDirName)
        rImagePath = path + "/" + rImageDirName

        rImageCount = 1
        # Read all images from a restaurant and create a feature + label mapped data structure
        rImages = getImageList(rImagePath)
        for image in rImages:

            featureLabel = getFeatures(rImageDirName,image, 'Trng')
            
            # Add all the features from feature vector to a common list, 
            if len(featureLabel) == ( featureVectorSize + 1):
                keyFeatureList.append(featureLabel)

                rImageCount += 1

    print "Feature Extraction Time: %s Seconds" % ( time.time() - fxStart )    

    trngFeature = []
    trngLabel = []

    # Shuffle Data
    np.random.shuffle(keyFeatureList)
    # Get all features in a array
    for featureAndLabel in keyFeatureList:
        #print "Y:", int(featureAndLabel[0])
        trngLabel.append(int(featureAndLabel[0]))
        trngFeature.append((np.delete(featureAndLabel, 0)).tolist())

    print "INFO: Train the model", len(trngFeature)
   
    trainStart = time.time()
    clf.fit(trngFeature,trngLabel)
    print "INFO: Training Time: %s seconds " % ( time.time() - trainStart )

      
'''
Function Name: getRestaurantName()
Input Args   : < Restaurant Index>
Return       : < Restaurant Name>
Description  : This function returns the name of the  Restaurant based on index value or label

'''
def getRestaurantName(rIndex):

    keys   = restaurantLabel.keys()
    for key in keys:
        if rIndex == restaurantLabel[key]:
            return key

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

'''
Function Name: getListOfDir()
Input Args   : < Path >
Return       : <Array: List of Directory >
Description  : This function returns all the directories under the specified paths
'''
def getListOfDir(path):
    # Read each sport action directory
    dirs  = os.listdir(path)
    #print dirs

    sportsActionsCount = 0
    filtered_dir  = []
    # Remove . .. and hidden directory
    for dir in dirs:
        if not dir.startswith("."):
            filtered_dir.append(dir)

    return filtered_dir



if __name__ == "__main__":
   
    sess.init_app(app)

    app.debug = True
    app.run()

   
        

