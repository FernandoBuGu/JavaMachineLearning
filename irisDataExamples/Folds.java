package irisDataExamples;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * Create folds for model evaluation with WEKA
 * 
 * 
 * Folds can be automatically created with crossValidateModel. 
 * Here, however, the folds are created manually in order to assess 
 * how performance varies across folds. 
 * A J48 is used to predict Species (categorical) on the Iris data.
 * For 3 folds, different results are printed using Evaluation from Weka.
 * This includes: Correctly Classified Instances, Root relative squared error...
 * Also, additional statistics are printed, like AUC or the root-relative-squared-error,
 * plus a confusion matrix.
 * 
 * @author feBueno, May 2020
 * fernando.bueno.gutie@gmail.com
 */

public class Folds {
	public static void main(String[] args) throws Exception {
		
		//load dataset
		DataSource source = new DataSource("src/data/irisTrain_df.arff");
		Instances data = source.getDataSet();
		
		//set class index to the last attribute (categorical)
		data.setClassIndex(data.numAttributes()-1);
		
		//build classifier
		J48 tree = new J48();

		//random
		Random rand = new Random(47);
		int folds=3;
		
		//randomly sort instances
		Instances randData = new Instances(data);
		randData.randomize(rand);
		if(randData.classAttribute().isNominal())
			randData.randomize(rand);
		
		//carry cross-validation
		for(int n=0;n<folds;n++) {
			Evaluation eval = new Evaluation(randData);
			//get folds
			Instances dataTrain = randData.trainCV(folds, n);
			Instances dataTest = randData.trainCV(folds, n);
			//build and evaluate classifier
			tree.buildClassifier(dataTrain);
			eval.evaluateModel(tree, dataTest);
			
			
			//print for each fold
			System.out.println(eval.toSummaryString("Evaluation results:\n",false));
			System.out.println("AUC: " + eval.areaUnderROC(1));
			System.out.println("RRSE:" + eval.rootRelativeSquaredError());
			System.out.println("Error Rate:" + eval.errorRate());
			System.out.println(eval.toMatrixString("=== Confusion Matrix ===\n"));

		}
	}
}/* Output
Evaluation results:

Correctly Classified Instances          65               98.4848 %
Incorrectly Classified Instances         1                1.5152 %
Kappa statistic                          0.9771
Mean absolute error                      0.0135
Root mean squared error                  0.0821
Relative absolute error                  3.029  %
Root relative squared error             17.3993 %
Total Number of Instances               66     

AUC: 0.9990384615384615
RRSE:17.39926923947753
Error Rate:0.015151515151515152
=== Confusion Matrix ===

  a  b  c   <-- classified as
 20  0  0 |  a = setosa
  0 25  1 |  b = versicolor
  0  0 20 |  c = virginica

Evaluation results:

Correctly Classified Instances          66               98.5075 %
Incorrectly Classified Instances         1                1.4925 %
Kappa statistic                          0.9774
Mean absolute error                      0.0189
Root mean squared error                  0.0971
Relative absolute error                  4.2453 %
Root relative squared error             20.6115 %
Total Number of Instances               67     

AUC: 0.9897959183673469
RRSE:20.611451697612022
Error Rate:0.014925373134328358
=== Confusion Matrix ===

  a  b  c   <-- classified as
 26  0  0 |  a = setosa
  0 18  0 |  b = versicolor
  0  1 22 |  c = virginica

Evaluation results:

Correctly Classified Instances          65               97.0149 %
Incorrectly Classified Instances         2                2.9851 %
Kappa statistic                          0.9552
Mean absolute error                      0.0366
Root mean squared error                  0.1353
Relative absolute error                  8.2382 %
Root relative squared error             28.6999 %
Total Number of Instances               67     

AUC: 0.9767676767676767
RRSE:28.69988849802639
Error Rate:0.029850746268656716
=== Confusion Matrix ===

  a  b  c   <-- classified as
 22  0  0 |  a = setosa
  0 20  2 |  b = versicolor
  0  0 23 |  c = virginica
  */
