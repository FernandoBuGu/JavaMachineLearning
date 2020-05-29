package basics;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * Create folds for model evaluation with WEKA
 * Folds can be automatically created with crossValidateModel, 
 * but here we control them, for instance to see how performance varies across folds,
 * or to save the folds for further computations
 * 
 * 
 * @author feBueno, May 2020
 * fernando.bueno.gutie@gmail.com
 */

public class Folds {
	public static void main(String[] args) throws Exception {
		
		//load dataset
		DataSource source = new DataSource("src/data/iris.arff");
		Instances data = source.getDataSet();
		
		//set class index to the 2nd last attribute (quantitative)
		data.setClassIndex(data.numAttributes()-1);
		
		//build classifier
		J48 tree = new J48();

		//random
		Random rand = new Random(47);
		int folds=10;
		
		//load test dataset
		DataSource sourceTest = new DataSource("src/data/irisTest.arff");
		Instances dataTest = sourceTest.getDataSet();
		dataTest.setClassIndex(dataTest.numAttributes()-1);
		
		
		Instances randData = new Instances(data);
		randData.randomize(rand);
		//stratify		
		if(randData.classAttribute().isNominal())
			randData.randomize(rand);
		
		//carry cross-validation
		for(int n=0;n<folds;n++) {
			Evaluation eval = new Evaluation(randData);
			//get folds
			Instances train = randData.trainCV(folds, n);
			Instances test = randData.trainCV(folds, n);
			//build and evaluate classifier
			tree.buildClassifier(train);
			eval.evaluateModel(tree, dataTest);
			
			
			//print for each fold
			System.out.println(eval.toSummaryString("Evaluation results:\n",false));
			System.out.println("AUC: " + eval.areaUnderROC(1));
			System.out.println("RRSE:" + eval.rootRelativeSquaredError());
			System.out.println("Error Rate:" + eval.errorRate());
			System.out.println(eval.toMatrixString("=== Confusion Matrix ===\n"));

		}
	}
}
