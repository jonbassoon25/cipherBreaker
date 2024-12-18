#main control file for all cipher breaker tools
import joblib

from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import MLRunner
import resultAnalysis
import cipherBreakerTrainer
import cipherBreaker
import modelAnalyzer
import breakerPredictor

#/------------------------------------------------------------------------------------------------/
#MLRunner
'''
#yTrain, xTrain, yTest, xTest = MLRunner.loadData("all")

#runML(xTrain, yTrain, xTest, yTest, NearestCentroid("euclidean"))
#runML(xTrain, yTrain, xTest, yTest, KNeighborsClassifier(n_neighbors=3, weights = "uniform"))
#runML(xTrain, yTrain, xTest, yTest, GaussianNB())
#runML(xTrain, yTrain, xTest, yTest, DecisionTreeClassifier(criterion="gini", splitter="best"))
#runML(xTrain, yTrain, xTest, yTest, RandomForestClassifier(n_estimators=5))
#runML(xTrain, yTrain, xTest, yTest, SVC(kernel="linear", C=1.0, degree = 3, gamma = "scale"))

#max run size < 1048576 before zsh: killed error. Not enough RAM
#multiRun(runs, clf(), "", 1, "compressed")

#short run range (2^6 - 2^11)

runs = [2 ** i for i in range(6, 12)]
MLRunner.multiRun([1024], DecisionTreeClassifier(criterion="gini", splitter="best"), "gini,best", 1, "compressed")
#multiRun(runs, MLPClassifier(hidden_layer_sizes=(32, 24), activation="relu", solver='lbfgs', alpha=1e-5, max_iter=10000), "(32,24),relu,lbfgs,1e-5,10000", 1, "compressed")
#multiRun(runs, MLPClassifier(hidden_layer_sizes=(100, 100), activation="relu", solver='lbfgs', alpha=1e-5, max_iter=10000), "(100, 100),relu,lbfgs,1e-5,10000", 1, "compressed")
#multiRun(runs, MLPClassifier(hidden_layer_sizes=(192, 140, 100), activation="relu", solver='lbfgs', alpha=1e-5, max_iter=1000000), "(192, 140, 100),relu,lbfgs,1e-5,1000000", 1, "compressed")
quit()
#medium run range (2^12 - 2^17)
runs = [2 ** i for i in range(12, 18)]


#log run range (2^18 - 2^20)
runs = [2 ** i for i in range(18, 20)]
MLRunner.multiRun(runs[1:], DecisionTreeClassifier(criterion="gini", splitter="best"), "gini,best", 1, "compressed")
MLRunner.multiRun(runs, DecisionTreeClassifier(criterion="entropy", splitter="best"), "entropy,best", 1, "compressed")
MLRunner.multiRun(runs, DecisionTreeClassifier(criterion="log_loss", splitter="best"), "log_loss,best", 1, "compressed")

MLRunner.multiRun(runs, DecisionTreeClassifier(criterion="gini", splitter="random"), "gini,random", 1, "compressed")
MLRunner.multiRun(runs, DecisionTreeClassifier(criterion="entropy", splitter="random"), "entropy,random", 1, "compressed")
MLRunner.multiRun(runs, DecisionTreeClassifier(criterion="log_loss", splitter="random"), "log_loss,random", 1, "compressed")

runs = [2 ** i for i in range(6, 10)]
MLRunner.multiRun(runs[2:], SVC(kernel="linear", C=1.0, degree = 3, gamma = "scale"), "linear,1.0,3,scale", 1, "compressed")
MLRunner.multiRun(runs, SVC(kernel="rbf", C=1.0, degree = 3, gamma = "scale"), "rbf,1.0,3,scale", 1, "compressed")
MLRunner.multiRun(runs, SVC(kernel="sigmoid", C=1.0, degree = 3, gamma = "scale"), "sigmoid,1.0,3,scale", 1, "compressed")
MLRunner.multiRun(runs, SVC(kernel="poly", C=1.0, degree = 3, gamma = "scale"), "poly,1.0,3,scale", 1, "compressed")

#exceptions / long run time sets

#estimated 20hrs per
runs = [2 ** i for i in range(12, 15)] #up to 16384:1
MLRunner.multiRun(runs, SVC(kernel="rbf", C=1.0, degree = 3, gamma = "scale"), "rbf,1.0,3,scale", trainingReps = 1, dataFormat = "uncompressed", dtype = "all")
MLRunner.multiRun(runs, SVC(kernel="sigmoid", C=1.0, degree = 3, gamma = "scale"), "sigmoid,1.0,3,scale", trainingReps = 1, dataFormat = "uncompressed", dtype = "all")
MLRunner.multiRun(runs, SVC(kernel="poly", C=1.0, degree = 3, gamma = "scale"), "poly,1.0,3,scale", trainingReps = 1, dataFormat = "uncompressed", dtype = "all")

#estimated 43hrs per
runs = [2 ** i for i in range(10, 12)] #up to 2048:1
MLRunner.multiRun(runs[1:], SVC(kernel="linear", C=1.0, degree = 3, gamma = "scale"), "linear,1.0,3,scale", 1, "compressed")
MLRunner.multiRun(runs, SVC(kernel="rbf", C=1.0, degree = 3, gamma = "scale"), "rbf,1.0,3,scale", 1, "compressed")
MLRunner.multiRun(runs, SVC(kernel="sigmoid", C=1.0, degree = 3, gamma = "scale"), "sigmoid,1.0,3,scale", 1, "compressed")
MLRunner.multiRun(runs, SVC(kernel="poly", C=1.0, degree = 3, gamma = "scale"), "poly,1.0,3,scale", 1, "compressed")
'''
#/------------------------------------------------------------------------------------------------/
#Result Analysis
#'''
#resultAnalysis.compareClassifierParameters("DecisionTreeClassifier", ["uncompressed"], "score", "all")
#resultAnalysis.compareClassifierParameters("GaussianNB", ["uncompressed"], "score", "all")
#resultAnalysis.compareClassifierParameters("KNeighborsClassifier", ["uncompressed"], "score", "all")
#resultAnalysis.compareClassifierParameters("NearestCentroid", ["uncompressed"], "score", "all")
#resultAnalysis.compareClassifierParameters("RandomForestClassifier", ["uncompressed"], "score", "all")
#resultAnalysis.compareClassifierParameters("SVC", ["uncompressed"], "score", "all")

#resultAnalysis.compareClassifiers(["DecisionTreeClassifier-entropy,random", "SVC-linear,1.0,3,scale"], "time", "all", "compressed")
#'''

#/------------------------------------------------------------------------------------------------/
#Cipher Breaker Trainer
'''
trainingDataToLetterRatio = 131072 * 2

clf, name, score, trainingTime = cipherBreakerTrainer.trainClf(DecisionTreeClassifier(criterion="gini", splitter="best"), trainingDataToLetterRatio, 0.05, "compressed", "all")

joblib.dump(clf, "./CCCs/cipherCharacterClassifier.pkl")
with open("./CCCs/cipherClassifierDescription.txt", "w") as CCDFile:
	CCDFile.write(f"{name} trained with {trainingDataToLetterRatio} values per letter\nScore of {round(score,3)}. Training Time of {round(trainingTime,2)}s")
'''

#/------------------------------------------------------------------------------------------------/
#Cipher Breaker
'''
trainingType = "uncompressed"
#charClassifier = joblib.load("./CCCs/cipherCharacterClassifier.pkl")
charClassifier = joblib.load(f"./CCCs/saved/{trainingType}/clf-3.pkl")
cipherBreaker.decryptUserMessages(charClassifier, trainingType)
'''

#/------------------------------------------------------------------------------------------------/
#Model Analyzer
'''
numTests = 5000 #number of times each character is tested for accuracy
trainingType = "uncompressed"
#charClassifier = joblib.load("./CCCs/cipherCharacterClassifier.pkl")
charClassifier = joblib.load(f"./CCCs/saved/{trainingType}/clf-3.pkl")

analysis = modelAnalyzer.analyze(charClassifier, trainingType, numTests, 0.0)
modelAnalyzer.plotAnalysis(analysis)
'''

#/------------------------------------------------------------------------------------------------/
#Breaker Predictor -- This one is out of date, use the Universal Cipher Breaker Breaker Predictor with ciphers.customCipher() or ciphers.uncompressedCustomCipher() as the encryptor
'''
trainingType = "uncompressed"
print("Loading Model...")
#clf = joblib.load("./CCCs/cipherCharacterClassifier.pkl")
clf = charClassifier = joblib.load(f"./CCCs/saved/{trainingType}/clf-2.pkl")

breakerPredictor.predictUserInput(clf, trainingType, includePredictedTextAsWords = False) #third option has to be true if orignal text contains non-words or unknown words
'''