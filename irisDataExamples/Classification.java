package irisDataExamples;


/**
 * Classification on Iris data using WEKA.
 * 
 * Prints failed predictions of Species (Iris data) on a test set.
 * A Logistic Model Tree (LMT) is used as classifier. 
 * 94% of samples are properly predicted with train and test sets of 100 and 50. 
 *  
 * Train and Test data was randomly sampled in R:
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
import weka.classifiers.trees.*;
import weka.classifiers.lazy.KStar;

import additionalClasses.LADTree;//import additional classes recognized by https://weka.sourceforge.io/doc.stable/ from github repositories

import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Classification {
	public static void main(String[] args) throws Exception {
		
		//load dataset
		DataSource source = new DataSource("src/data/irisTrain_df.arff");
		Instances data = source.getDataSet();
		
		//set class index to the last attribute (categorical)
		data.setClassIndex(data.numAttributes()-1);
		
		//get number of classes and names
		int numClasses = data.numClasses();
		for(int i=0;i<numClasses;i++) {
			String classValue = data.classAttribute().value(i);
			System.out.println("class value "+i+ " is: "+classValue);
		}
		
		//build classifier
		LMT tree = new LMT();//weka.classifiers.Classifier
		//LMT: logistic model trees, which are classification trees with logistic regression functions at the leaves.
		tree.buildClassifier(data);
		
		//load test dataset
		DataSource sourceTest = new DataSource("src/data/irisTest_df.arff");
		Instances dataTest = sourceTest.getDataSet();
		dataTest.setClassIndex(dataTest.numAttributes()-1);
		
		//carry and print predictions
		int failed=0;
		for(int n=0;n<dataTest.numInstances();n++) {
			
			double actualClass = dataTest.instance(n).classValue();
			String actual = dataTest.classAttribute().value((int)actualClass);
			Instance newInst = dataTest.instance(n);
			double predTree = tree.classifyInstance(newInst);
			String predString = dataTest.classAttribute().value((int)predTree);//labels are automatically hidden
			//print only failed classifications
			if(!actual.equals(predString)) {
				System.out.print("failed: ");
				failed++;
				System.out.println(actual+", "+predString);
			}
		}
		System.out.println("Label of "+ (100-(failed*100/dataTest.numInstances()))+" % instances was properly predicted for " + (data.attribute(data.classIndex()).name()));
	}
} /*Output:
	class value 0 is: setosa
	class value 1 is: versicolor
	class value 2 is: virginica
	failed: versicolor, virginica
	failed: versicolor, virginica
	faile:, virginica, versicolor
	Label of 94 % instances was properly predicted for Species
*/
