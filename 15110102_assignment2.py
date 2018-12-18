dataset = int(input('Enter 1 for MNIST/ 2 for Credit Card: '))
algo = int(input('Enter 1 for K-nearest neighbour/ 2 for Decision Tree/ 3 for SVM and 4 for Logistic Regression: '))
import matplotlib.pyplot as plt
from sklearn import tree
import seaborn as sns
from sklearn import datasets, svm, metrics
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from numpy import genfromtxt
data1 = genfromtxt('UCI_Credit_Card.csv', delimiter = ',')[1:]
ddata = data1[:,:24]

dlabel = data1[:,24]



def Score(mat,dataset):
    s = 0
    if dataset == 1:
        l = 10
    else:
        l = 2
    for j in range(0,l):
        s = s + mat[j,j]

    score = (s/np.sum(mat))*100
    return score
def Print(classifier,expected,predicted):
    print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

def T3_Score(classifier,x_test,y_test):
    s = 0
    for i in range(0,len(x_test)):
        c = classifier.predict_proba(x_test)[i]
        a1 = np.argmax(c)
        c[a1] = 0
        a2 = np.argmax(c)
        c[a2] = 0
        a3 = np.argmax(c)
        if y_test[i] == a1 or y_test[i] == a2 or y_test[i] == a3 :
            s = s +1
    return (s/len(x_test))*100
def ROC(classifier,x_test,y_test):
    probs = classifier.predict_proba(x_test)
 

    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show(block = False)


def Algo(dataset,algo,x_train, x_test, y_train, y_test,digits):
    
    if algo == 1:
        classifier =  KNeighborsClassifier(n_neighbors = 3)
        
    if algo == 2:
        classifier = tree.DecisionTreeClassifier()
        
    if algo == 4:
        classifier = LogisticRegression()
    if dataset == 1:
        if algo == 3:
            random_state = np.random.RandomState(0)
            classifier = svm.SVC(kernel='poly', probability=True,random_state=random_state)
    elif dataset == 2:
        if algo == 3:
            classifier = svm.LinearSVC()
   
 
    plt.show(block = False)
          
        
        
    classifier.fit(x_train,y_train)
   
    expected = y_test
    predicted = classifier.predict(x_test)
 
    mat = metrics.confusion_matrix(expected, predicted)

    accuracy = Score(mat,dataset)

    print('Accuracy',accuracy)
    Print(classifier,expected,predicted)
    
   
        
        
    if dataset == 1 :
        print('T1_score:',accuracy)
        
        
        T3score = T3_Score(classifier,x_test,y_test)

       
        
        print('T3_score:',T3score)
    if dataset == 2 and algo !=3:
        ROC(classifier,x_test,y_test)
    if dataset == 1:
        data2 = digits.data[0:4]
        predicteds = classifier.predict(data2)
        plt.suptitle("MNIST Dataset", fontsize=13)
        
        images_and_labels = list(zip(digits.images, digits.target))
        for index, (image, label) in enumerate(images_and_labels[:4]):
            plt.subplot(2, 4, index + 1)
            plt.axis('off')
            plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
            plt.title('Training: %i' % label)
        images_and_predictions = list(zip(digits.images, predicteds))
        for index, (image, prediction) in enumerate(images_and_predictions[:4]):
            plt.subplot(2, 4, index + 5)
            plt.axis('off')
            plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
            plt.title('Prediction: %i' % prediction)
        

        plt.show(block = False)
    
    
   
    

    
    plt.figure(figsize=(9,9))
    sns.heatmap(mat, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {:.3f}'.format(accuracy) + '%'
    plt.title(all_sample_title, size = 15)
    plt.show(block = False)
    
            


def Main(dataset,algo):
    if dataset == 1:
        digits = datasets.load_digits()
        x_train, x_test, y_train, y_test = train_test_split(digits.data[4:], digits.target[4:], test_size=0.20, random_state=0)
        Algo(dataset,algo,x_train, x_test, y_train, y_test,digits)
    else:
        digits = datasets.load_digits()
        x_train, x_test, y_train, y_test = train_test_split(ddata, dlabel, test_size=0.20, random_state=0)
        Algo(dataset,algo,x_train, x_test, y_train, y_test,digits)
        

Main(dataset,algo)
plt.show()

