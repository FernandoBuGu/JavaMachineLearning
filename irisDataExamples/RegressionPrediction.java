package irisDataExamples;

/**
 * Regression predictions using WEKA. 
 * 
 * Support Vector Machine regression is applied on the Iris data, and predictions are printed.
 * 94.588% instances were cached with a 95% Confidence Interval.
 * 
 *  Train and Test data was randomly sampled in R from Iris dataset:
 *		irisTrain_df=iris[sample(nrow(iris), 100), ]
 *		irisTest_df=iris[!rownames(iris) %in% rownames(irisTrain_df),]
 *		library(RWeka)
 *		setwd("/home/febueno/eclipse-workspace/JavaWeka/src/data")
 *		write.arff(irisTrain_df, file = "irisTrain_df.arff")
 *		write.arff(irisTest_df, file = "irisTest_df.arff")
 * 
 * REQUIRE:
 * Depending on Weka installation, it may be required to add arpack_combined.jar core.jar and mtj.jar from
 * https://svn.cms.waikato.ac.nz/svn/weka/branches/stable-3-8/weka/lib/
 * 
 * @author feBueno, May 2020
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
import weka.classifiers.functions.SMOreg;
import weka.classifiers.trees.*;
import weka.classifiers.lazy.KStar;

import additionalClasses.LADTree;//import additional classes recognized by https://weka.sourceforge.io/doc.stable/ from github repositories

import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class RegressionPrediction {
	public static void main(String[] args) throws Exception {
		
		//load dataset
		DataSource source_DS = new DataSource("src/data/irisTrain_df.arff");
		Instances data_intances = source_DS.getDataSet();
		
		//set class index to the last attribute (quantitative)
		data_intances.setClassIndex(data_intances.numAttributes()-2);
		
		//build model
		SMOreg SMOr_model = new SMOreg();//weka.classifiers.functions.SMOreg
		SMOr_model.buildClassifier(data_intances);
		System.out.println("=====support vector machine regression=====");
		System.out.println(SMOr_model);
		
		//load test dataset
		DataSource sourceTest_DS = new DataSource("src/data/irisTest_df.arff");
		Instances dataTest_instances = sourceTest_DS.getDataSet();
		dataTest_instances.setClassIndex(dataTest_instances.numAttributes()-2);
		
		//handle decimals
		DecimalFormat f = new DecimalFormat("##.00");
		
		//carry and print predictions
		System.out.println("Actual Class, SMO Predicted");
		for(int n=0;n<dataTest_instances.numInstances();n++) {
			
			double actualValue_dou = dataTest_instances.instance(n).classValue();
			
			Instance newInst_instance = dataTest_instances.instance(n);
			
			double predSMOr_dou = SMOr_model.classifyInstance(newInst_instance);
			
			System.out.println(f.format(actualValue_dou)+", "+f.format(predSMOr_dou));
		}
	}
} /* Output:
=====support vector machine regression=====
SMOreg

weights (not support vectors):
 -       0.0973 * (normalized) Sepal.Length
 +       0.2327 * (normalized) Sepal.Width
 +       0.3558 * (normalized) Petal.Length
 -       0.3091 * (normalized) Species=setosa
 +       0.0493 * (normalized) Species=versicolor
 +       0.2598 * (normalized) Species=virginica
 +       0.2351



Number of kernel evaluations: 5050 (94.588% cached)
Actual Class, SMO Predicted
.10, .17
.30, .24
.30, .32
.20, .24
.40, .31
.20, .24
.20, .15
.20, .25
.20, .23
.40, .21
.10, .40
.20, .17
.20, .14
.10, .28
.30, .23
.20, .30
1.50, 1.41
1.50, 1.32
1.40, 1.27
1.00, 1.05
1.50, 1.41
1.00, 1.26
1.50, 1.17
1.10, 1.20
1.40, 1.33
1.70, 1.42
1.00, 1.15
1.60, 1.40
1.60, 1.49
1.30, 1.17
1.30, 1.22
1.30, 1.29
1.30, 1.32
2.50, 2.18
2.10, 2.03
1.70, 1.84
2.00, 2.00
2.00, 1.86
1.80, 2.01
2.30, 2.04
1.80, 1.86
1.60, 2.01
2.00, 2.26
2.20, 1.98
1.50, 1.91
2.40, 2.14
1.80, 1.94
2.40, 2.04
2.30, 2.10
1.80, 1.99
*/
