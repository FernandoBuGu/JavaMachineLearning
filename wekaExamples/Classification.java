package basics;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * Classification on Iris data using WEKA.
 * 
 * 
 * @author feBueno, May 2020
 * fernando.bueno.gutie@gmail.com
 */

public class Classification {
	public static void main(String[] args) throws Exception {
		
		//load dataset
		DataSource source = new DataSource("src/data/iris.arff");
		Instances data = source.getDataSet();
		
		//set class index to the 2nd last attribute (quantitative)
		data.setClassIndex(data.numAttributes()-1);
		
		//get number of classes
		int numClasses = data.numClasses();
		for(int i=0;i<numClasses;i++) {
			String classValue = data.classAttribute().value(i);
			System.out.println("class value "+i+ "is: "+classValue);
		}
		
		//build classifier
		J48 tree = new J48();
		tree.buildClassifier(data);
		
		//load test dataset
		DataSource sourceTest = new DataSource("src/data/irisUnknown.arff");//iris dataset with hidden labels
		Instances dataTest = sourceTest.getDataSet();
		dataTest.setClassIndex(dataTest.numAttributes()-1);
		

		
		/*Instances randData = new Instances(data);
		randData.randomize(rand);
		//stratify		
		if(randData.classAttribute().isNominal())
			randData.randomize(rand);*/
		
		//carry predictions
		for(int n=0;n<dataTest.numInstances();n++) {
			
			double actualClass = dataTest.instance(n).classValue();
			String actual = dataTest.classAttribute().value((int)actualClass);
			Instance newInst = dataTest.instance(n);
			double predTree = tree.classifyInstance(newInst);
			String predString = dataTest.classAttribute().value((int)predTree);
			//print for each instance
			System.out.println(actual+", "+predString);
		}
	}
}
