package irisDataExamples;

/**
 * Combining classifier models in WEKA. 
 * 
 * Multiple meta-base-classifiers are defined, 
 * plus the stacking-base classifier is used to combine multiple base-classifiers.
 * After fitting an Iris train dataset in these, the models are saved in src/models
 * 
 * @author feBueno, June 2020
 * fernando.bueno.gutie@gmail.com
 */

import java.text.DecimalFormat;

/**
 * Prediction with regression on Iris data using WEKA.
 * 
 * Prints failed predictions of Species (Iris data) on a test set.
 * A Logistic Model Tree (LMT) is used as classifier. 
 * 94% of samples are properly predicted with train and test sets of 100 and 50. 
 *  
 * Train and Test data was randomly sampled in R from the Iris dataset:
 *	irisTrain_df=iris[sample(nrow(iris), 100), ]
 *	irisTest_df=iris[!rownames(iris) %in% rownames(irisTrain_df),]
 *	library(RWeka)
 *	setwd("/home/febueno/eclipse-workspace/JavaWeka/src/data")
 *	write.arff(irisTrain_df, file = "irisTrain_df.arff")
 *	write.arff(irisTest_df, file = "irisTest_df.arff")
 * 
 * @author feBueno, May 2020
 * fernando.bueno.gutie@gmail.com
 */

import java.util.Random;

//some classifiers accessible directly from weka.jar
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.trees.*;
import weka.classifiers.trees.lmt.LogisticBase;
import weka.classifiers.lazy.KStar;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.CostSensitiveClassifier;
import weka.classifiers.meta.LogitBoost;
import weka.classifiers.meta.MultiBoostAB;
import weka.classifiers.meta.MultiClassClassifier;
import weka.classifiers.meta.Stacking;
import additionalClasses.LADTree;//import additional classes recognized by https://weka.sourceforge.io/doc.stable/ from github repositories
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Ensembles {
	public static void main(String[] args) throws Exception {
		
		//load dataset
		DataSource source_DS = new DataSource("src/data/irisTrain_df.arff");
		Instances data_intances = source_DS.getDataSet();
		
		//set class index to the last attribute (categorical)
		data_intances.setClassIndex(data_intances.numAttributes()-1);
		
		//In each ensamble, a meta classifier is combined with a base classifier
		
		//ensamble, approach 1: multi-class logistic regression
		MultiClassClassifier MultiClassClassifier_model = new MultiClassClassifier();//meta: handling multi-class datasets with 2-class classifiers
		MultiClassClassifier_model.setClassifier(new Logistic());//base: logistic regression
		MultiClassClassifier_model.buildClassifier(data_intances);
		
		//ensamble, approach 2: additive logistic regression, can handle multi-class
		LogitBoost LogitBoost_model = new LogitBoost();//meta: additive logistic regression
		LogitBoost_model.setClassifier(new LogisticBase());//base: logistic regression for LogitBoost
		LogitBoost_model.buildClassifier(data_intances);
		
		//ensamble, approach 3: boost a classifier, improve performance at expenses of overfit
		AdaBoostM1 Boost_model = new AdaBoostM1();//meta
		Boost_model.setClassifier(new BayesNet());//base
		Boost_model.setNumIterations(100);
		Boost_model.buildClassifier(data_intances);
		
		//ensamble, approach 4: bagging a classifier to reduce variance
		Bagging Bagging_model = new Bagging();//meta
		Bagging_model.setClassifier(new RandomForest());
		Bagging_model.buildClassifier(data_intances);
		
		//ensamble, approach 5: stacking multiple base-classifiers
		Stacking stacker_model = new Stacking(); 
		stacker_model.setMetaClassifier(new Logistic());
		Classifier[] classifiers_modelArr = {new J48(), new NaiveBayes(), new RandomForest()};
		stacker_model.setClassifiers(classifiers_modelArr);
		stacker_model.buildClassifier(data_intances);
		
		//save models as binary files
		weka.core.SerializationHelper.write("src/models/MultiClassClassifier_model", MultiClassClassifier_model);
		weka.core.SerializationHelper.write("src/models/LogitBoost_model", LogitBoost_model);
		weka.core.SerializationHelper.write("src/models/Boost_model", Boost_model);
		weka.core.SerializationHelper.write("src/models/Bagging_model", Bagging_model);
		weka.core.SerializationHelper.write("src/models/stacker_model_OnRawTainData_model", stacker_model);
		
		System.out.println("Ensembl models saved in src/models");
		
	}
} /* Output:
Ensembl models saved in src/models
*/
