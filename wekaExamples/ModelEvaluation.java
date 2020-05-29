package basics;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * Model evaluation with WEKA
 * 
 * 
 * 
 * @author feBueno, May 2020
 * fernando.bueno.gutie@gmail.com
 */

public class ModelEvaluation {
	public static void main(String[] args) throws Exception {
		
		//load dataset
		DataSource source = new DataSource("src/data/iris.arff");
		Instances data = source.getDataSet();
		
		//set class index to the 2nd last attribute (quantitative)
		data.setClassIndex(data.numAttributes()-1);
		
		//build classifier
		J48 tree = new J48();
		tree.buildClassifier(data);

		//evaluation
		Evaluation eval = new Evaluation(data);
		Random rand = new Random(47);
		int folds=10;
		
		//load test dataset
		DataSource sourceTest = new DataSource("src/data/irisTest.arff");
		Instances dataTest = sourceTest.getDataSet();
		dataTest.setClassIndex(dataTest.numAttributes()-1);
		
		//evaluate model
		//eval.evaluateModel(tree, dataTest);
		eval.crossValidateModel(tree, dataTest, folds, rand);//if nominal, carries stratification
		
		System.out.println(eval.toSummaryString("Evaluation results:\n",false));
		System.out.println("AUC: " + eval.areaUnderROC(1));
		System.out.println("RRSE:" + eval.rootRelativeSquaredError());
		System.out.println("Error Rate:" + eval.errorRate());
		System.out.println(eval.toMatrixString("=== Confusion Matrix ===\n"));


	}
}
