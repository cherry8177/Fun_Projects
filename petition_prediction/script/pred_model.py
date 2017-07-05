import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model, decomposition, datasets
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def struct_train_test_feature(df):
	#assign features and response
	feature_list=['Title_str_len', 'Text_len_p1', 'Text_len_p2',
                  'Title_len', 'Text_str_len','Image','Tweet']
	x, y = (df[feature_list], 
         df.No_Supporters_log)

    #split train and test
	x_train, x_test, y_train, y_test = train_test_split(
	    x, y, test_size=0.25, random_state=42)

	#features for text and title
	text_court_vectorizer = CountVectorizer(analyzer = "word",   
                             tokenizer = None,
                             ngram_range=(1, 2),
                             preprocessor = None, 
                             stop_words = None,   
                             max_features = 2500) 
    train_text_features = text_court_vectorizer.fit_transform((x_train.Text).values.astype('U'))
	train_text_features = train_text_features.toarray()
	test_text_features = text_court_vectorizer.transform((x_test.Text).values.astype('U'))
	test_text_features = test_text_features.toarray()

	title_court_vectorizer = CountVectorizer(analyzer = "word",   
                             tokenizer = None,
                             ngram_range=(1, 3),
                             preprocessor = None, 
                             stop_words = None,   
                             max_features = 2500) 
    train_title_features = title_court_vectorizer.fit_transform((x_train.Title).values.astype('U'))
	train_title_features = train_title_features.toarray()
	test_title_features = title_court_vectorizer.transform((x_test.Title).values.astype('U'))
	test_title_features = test_title_features.toarray()

	total_scale_train=StandardScaler().fit(np.concatenate((train_text_features,
                                                     train_title_features,x_train[feature_list]), axis=1))	

	total_train_scale=total_scale_train.transform(np.concatenate((train_text_features,
	                                                     train_title_features,x_train[
	                                                         feature_list]),axis=1))

	total_test_scale=total_scale_train.transform(np.concatenate((test_text_features,
	                                                             test_title_features, 
	                                                x_test[feature_list]),axis=1))
	#PCA for dimention reduction
	train_pca=PCA(n_components=1500,svd_solver='full')
	x_train_pca=train_pca.fit_transform(total_train_scale)
	x_test_pca=train_pca.transform(total_test_scale)

	return x_train_pca, x_test_pca, y_train, y_test



	

def model_selection(x_train_pca, y_train):
	Elastic = linear_model.ElasticNet()
	pca = decomposition.PCA()
	pipe = Pipeline(steps=[('pca', pca), 
	                       ('Elastic', Elastic)])

	n_components = [ 60,100,500,750, 1000,]
	alphas = np.logspace(-4, 4, 4)
	l1_ratio=( 0.2,0.4,0.6,0.7,0.8,1)

	estimator = GridSearchCV(pipe,
	                         dict(pca__n_components=n_components,
	                              Elastic__alpha=alphas,Elastic__l1_ratio=l1_ratio))

	estimator.fit(x_train_pca, y_train)
	n_comp=estimator.best_estimator_.named_steps['pca'].n_components
	alpha=estimator.best_estimator_.named_steps['Elastic'].alpha
	l1=estimator.best_estimator_.named_steps['Elastic'].l1_ratio

	print("n_component:", n_comp,'\n', "alpha:", alpha,'\n',"l1_ratio: "l1)

	Elastic=linear_model.ElasticNet(alpha=alpha, 
                    l1_ratio=l1, fit_intercept=True, 
            normalize=False, precompute=False, max_iter=1000, 
           copy_X=True, tol=0.0001, warm_start=False, positive=False, 
           random_state=None, selection='cyclic')

	train_Elastic=Elastic.fit(x_train_pca[:,0:n_comp],y_train)
	train_predict=train_Elastic.predict(x_train_pca[:,0:n_comp])
	test_predict=train_Elastic.predict(x_test_pca[:,0:n_comp])
	MSE_train=mean_squared_error(y_train, train_predict)
	MSE_test=mean_squared_error(y_test, test_predict)
	test_r2=r2_score(y_test, test_predict)
	print("train error:", MSE_train,"\n", "test error:", MSE_test,"\n", "test r-sqaure:", test_r2 )
	return MSE_test



















					