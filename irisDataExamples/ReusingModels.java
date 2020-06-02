package irisDataExamples;

/**
 * Reusing models in WEKA. 
 * 
 * A MultilayerPerceptron model is saved as binary and then loaded to make and
 * print predictions in a test Iris dataset.
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
import weka.classifiers.lazy.KStar;

import additionalClasses.LADTree;//import additional classes recognized by https://weka.sourceforge.io/doc.stable/ from github repositories

import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class ReusingModels {
	public static void main(String[] args) throws Exception {
		
		//load dataset
		DataSource source_DS = new DataSource("src/data/irisTrain_df.arff");
		Instances data_intances = source_DS.getDataSet();
		
		//set class index to the last attribute (quantitative)
		data_intances.setClassIndex(data_intances.numAttributes()-2);
		
		//build model
		MultilayerPerceptron MLP_model = new MultilayerPerceptron();//weka.classifiers.functions.SMOreg
		MLP_model.buildClassifier(data_intances);
		System.out.println("=====multilayer perceptron regression=====");
		System.out.println(MLP_model);
		
		//save model
		weka.core.SerializationHelper.write("src/models/MultilayerPerceptron_OnRawTainData_model", MLP_model);//saved as binary file
		System.out.println("Model saved in src/models/MultilayerPerceptron_OnRawTrainData_model");
		
		
		//load model
		MultilayerPerceptron MLP_loaded_model = (MultilayerPerceptron) weka.core.SerializationHelper.read("src/models/MultilayerPerceptron_OnRawTainData_model");
		Evaluation eval = new Evaluation(data_intances);
		eval.evaluateModel(MLP_loaded_model, data_intances);
		System.out.println(eval.toSummaryString());
		
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
			
			double predSMOr_dou = MLP_loaded_model.classifyInstance(newInst_instance);
			
			System.out.println(f.format(actualValue_dou)+", "+f.format(predSMOr_dou));
		}
	}
} /* Output:
=====multilayer perceptron regression=====
Linear Node 0
    Inputs    Weights
    Threshold    0.26307556857769415
    Node 1    0.8457958134873012
    Node 2    -1.3298307901136788
    Node 3    -0.7790379565100111
Sigmoid Node 1
    Inputs    Weights
    Threshold    -1.318602356033643
    Attrib Sepal.Length    0.6020892062844625
    Attrib Sepal.Width    3.5828606289804337
    Attrib Petal.Length    -0.583760878866813
    Attrib Species=setosa    -1.579586247915816
    Attrib Species=versicolor    0.8661878931042087
    Attrib Species=virginica    1.9319558485296524
Sigmoid Node 2
    Inputs    Weights
    Threshold    -1.1274773079726146
    Attrib Sepal.Length    -0.02208494410219749
    Attrib Sepal.Width    -0.7529537448517662
    Attrib Petal.Length    -1.8986001947295883
    Attrib Species=setosa    1.2968204099492715
    Attrib Species=versicolor    0.38510042150972623
    Attrib Species=virginica    -0.5455505458885357
Sigmoid Node 3
    Inputs    Weights
    Threshold    -1.596440503025026
    Attrib Sepal.Length    0.8494254555034558
    Attrib Sepal.Width    1.2468786265854983
    Attrib Petal.Length    0.21379991027393846
    Attrib Species=setosa    0.6116923733533196
    Attrib Species=versicolor    0.6711342531910003
    Attrib Species=virginica    0.33130871857417316
Class 
    Input
    Node 0

Model saved in src/models/MultilayerPerceptron_OnRawTrainData_model

Correlation coefficient                  0.9836
Mean absolute error                      0.1072
Root mean squared error                  0.1461
Relative absolute error                 16.3651 %
Root relative squared error             19.2879 %
Total Number of Instances              100     

Actual Class, SMO Predicted
.10, .15
.30, .16
.30, .20
.20, .18
.40, .19
.20, .14
.20, .15
.20, .17
.20, .17
.40, .15
.10, .24
.20, .15
.20, .12
.10, .17
.30, .15
.20, .18
1.50, 1.51
1.50, 1.34
1.40, 1.21
1.00, .93
1.50, 1.40
1.00, 1.24
1.50, 1.20
1.10, 1.15
1.40, 1.39
1.70, 1.45
1.00, 1.09
1.60, 1.38
1.60, 1.64
1.30, 1.20
1.30, 1.17
1.30, 1.26
1.30, 1.34
2.50, 2.25
2.10, 2.09
1.70, 1.68
2.00, 2.23
2.00, 1.71
1.80, 2.08
2.30, 1.79
1.80, 1.86
1.60, 2.10
2.00, 2.30
2.20, 1.92
1.50, 1.93
2.40, 2.32
1.80, 2.08
2.40, 2.16
2.30, 2.21
1.80, 2.06
*/
