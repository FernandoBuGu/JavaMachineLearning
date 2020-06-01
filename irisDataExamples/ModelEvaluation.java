package irisDataExamples;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * Model evaluation with WEKA
 *  
 *  
 * A J48 is used to predict Species (categorical) on the Iris data using cross-validation. 
 * Results are printed using Evaluation from Weka.
 * This includes: Correctly Classified Instances, Root relative squared error...
 * Also, additional statistics are printed, like AUC or the root-relative-squared-error,
 * plus a confusion matrix.
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
		DataSource sourceTest = new DataSource("src/data/irisTest_df.arff");
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
} /* Output:
Evaluation results:

Correctly Classified Instances          45               90      %
Incorrectly Classified Instances         5               10      %
Kappa statistic                          0.8499
Mean absolute error                      0.0775
Root mean squared error                  0.2465
Relative absolute error                 17.3725 %
Root relative squared error             52.0853 %
Total Number of Instances               50     

AUC: 0.9278074866310161
RRSE:52.08527549435704
Error Rate:0.1
=== Confusion Matrix ===

  a  b  c   <-- classified as
 16  0  0 |  a = Iris-setosa
  0 15  2 |  b = Iris-versicolor
  0  3 14 |  c = Iris-virginica
*/
