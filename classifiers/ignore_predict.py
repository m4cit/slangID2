import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB   
from .log_regression import dm_log_reg, dm_train_log
from .linear_svm import dm_lin_svm, dm_train_svm
from .decision_tree import dm_dec_tree, dm_train_dt
from .gaussian_nb import dm_naive_g, dm_train_nbg
from .multinomial_nb import dm_naive_m, dm_train_nbm
import pickle

################ splitting the data #########################################################
data = pd.read_csv (r'./classifiers/modifiers/data/handpicked_data.csv')
train, test = train_test_split(data, test_size=.09, random_state=42, stratify=data['type'])
train_x = [x for x in train['phrase']]
train_y = [x for x in train['type']]
test_x = [x for x in test['phrase']]
test_y = [x for x in test['type']]

vectorizer = CountVectorizer(ngram_range=(1,2))
vect_train_x = vectorizer.fit_transform(train_x)
vect_test_x = vectorizer.transform(test_x)

############## functions to train classifiers ################

def train_svm():
    clf_svm = svm.SVC(kernel='linear')
    clf_svm.fit(vect_train_x, train_y)
    # saving trained models
    with open('./classifiers/models/pd_linear_svm_classifier.pkl', 'wb') as f:
        pickle.dump(clf_svm,f)
    print('\nLinear SVM trained...\n')

def train_log():
    clf_log = LogisticRegression(C=32, fit_intercept=False, solver='newton-cg')
    clf_log.fit(vect_train_x, train_y)
    # saving trained models
    with open('./classifiers/models/pd_log_regression_classifier.pkl', 'wb') as f:
        pickle.dump(clf_log,f)
    print('\nLogistic Regression trained...\n')

def train_dt():
    clf_dt = DecisionTreeClassifier(criterion='log_loss', min_samples_split=2)
    clf_dt.fit(vect_train_x, train_y)
    # saving trained models
    with open('./classifiers/models/pd_decision_tree_classifier.pkl', 'wb') as f:
        pickle.dump(clf_dt,f)
    print('\nDecision Tree trained...\n')

def train_nbg():
    clf_nbg = GaussianNB()
    clf_nbg.fit(vect_train_x.toarray(), train_y)
    # saving trained models
    with open('./classifiers/models/pd_gaussian_nb_classifier.pkl', 'wb') as f:
        pickle.dump(clf_nbg,f)
    print('\nNaive Bayes (Gaussian) trained...\n')

def train_nbm():
    clf_nbm = MultinomialNB(alpha=2.0, fit_prior=False)
    clf_nbm.fit(vect_train_x, train_y)
    # saving trained models
    with open('./classifiers/models/pd_multinomial_nb_classifier.pkl', 'wb') as f:
        pickle.dump(clf_nbm,f)
    print('\nNaive Bayes (Multinomial) trained...\n')

####################################################################################

############# following functions to predict the input phrase, as well as print the mean accuracy of the classifier##############

def predictor_svm(input):
    try:
        # loading pre-trained classifier
        with open('./classifiers/models/pd_linear_svm_classifier.pkl','rb') as f:
            loaded_clf = pickle.load(f)
        print("\n Linear SVM:\n Phrase:      ", input[0], "Prediction:    ", loaded_clf.predict(vectorizer.transform(input))[0])
        def test_score():
            print(" mean accuracy: ", loaded_clf.score(vect_test_x, test_y))
        test_score()
    except FileNotFoundError:
        print("Please train the model(s) first!")


def predictor_log(input):
    try:
        # loading pre-trained classifier
        with open('./classifiers/models/pd_log_regression_classifier.pkl','rb') as f:
            loaded_clf = pickle.load(f)
        print("\n Logistic Regression:\n Phrase:      ", input[0], "Prediction:    ", loaded_clf.predict(vectorizer.transform(input))[0])
        def test_score():
            print(" mean accuracy: ", loaded_clf.score(vect_test_x, test_y))
        test_score()
    except FileNotFoundError:
        print("Please train the model(s) first!")


def predictor_dec(input):
    try:
        # loading pre-trained classifier
        with open('./classifiers/models/pd_decision_tree_classifier.pkl','rb') as f:
            loaded_clf = pickle.load(f)
        print("\n Decision Tree:\n Phrase:      ", input[0], "Prediction:    ", loaded_clf.predict(vectorizer.transform(input))[0])
        def test_score():
            print(" mean accuracy: ", loaded_clf.score(vect_test_x, test_y))
        test_score()
    except FileNotFoundError:
        print("Please train the model(s) first!")
    

def predictor_naive_g(input):
    try:
        # loading pre-trained classifier
        with open('./classifiers/models/pd_gaussian_nb_classifier.pkl','rb') as f:
            loaded_clf = pickle.load(f)
        print("\n Naive Bayes (Gaussian):\n Phrase:      ", input[0], "Prediction:    ", loaded_clf.predict(vectorizer.transform(input).toarray())[0])
        def test_score():
            print(" mean accuracy: ", loaded_clf.score(vect_test_x.toarray(), test_y))
        test_score()
    except FileNotFoundError:
        print("Please train the model(s) first!")
    

def predictor_naive_m(input):
    try:
        # loading pre-trained classifier
        with open('./classifiers/models/pd_multinomial_nb_classifier.pkl','rb') as f:
            loaded_clf = pickle.load(f)
        print("\n Naive Bayes (Multinomial):\n Phrase:      ", input[0], "Prediction:    ", loaded_clf.predict(vectorizer.transform(input))[0])
        def test_score():
            print(" mean accuracy: ", loaded_clf.score(vect_test_x, test_y))
        test_score()
    except FileNotFoundError:
        print("Please train the model(s) first!")
    

# training all models
def train_all():
    train_svm()
    train_dt()
    train_nbg()
    train_nbm()
    train_log()

# running all models
def predictor_all(input):
    predictor_svm(input)
    predictor_dec(input)
    predictor_naive_g(input)
    predictor_naive_m(input)
    predictor_log(input)

# training all demo models
def dm_train_all():
    dm_train_svm()
    dm_train_dt()
    dm_train_nbg()
    dm_train_nbm()
    dm_train_log()

# running all demo models
def dm_predict_all():
    dm_lin_svm()
    dm_dec_tree()
    dm_naive_g()
    dm_naive_m()
    dm_log_reg()
