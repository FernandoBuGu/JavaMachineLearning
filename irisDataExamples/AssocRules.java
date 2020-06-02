package irisDataExamples;

/**
 * Association rules in WEKA. 
 * 
 * Print the best associations between variables in the weatherNominal dataset using an Apriori-type algorithm
 * 
 * @author feBueno, June 2020
 * fernando.bueno.gutie@gmail.com
 */

import java.text.DecimalFormat;

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
import weka.associations.Apriori;
import weka.associations.AprioriItemSet;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class AssocRules {
	public static void main(String[] args) throws Exception {
		
		//load dataset
		DataSource source_DS = new DataSource("src/data/weatherNominal.arff");//all attributes should be nominal
		Instances data_intances = source_DS.getDataSet();
		
		//set class index to the last attribute (categorical)
		data_intances.setClassIndex(data_intances.numAttributes()-1);
		
		Apriori Apriori_model = new Apriori();//https://weka.sourceforge.io/doc.stable/
		Apriori_model.buildAssociations(data_intances);
		
		System.out.println(Apriori_model);
		
	}
} /* Output:
Apriori
=======

Minimum support: 0.15 (2 instances)
Minimum metric <confidence>: 0.9
Number of cycles performed: 17

Generated sets of large itemsets:

Size of set of large itemsets L(1): 12

Size of set of large itemsets L(2): 47

Size of set of large itemsets L(3): 39

Size of set of large itemsets L(4): 6

Best rules found:

 1. outlook=overcast 4 ==> play=yes 4    <conf:(1)> lift:(1.56) lev:(0.1) [1] conv:(1.43)
 2. temperature=cool 4 ==> humidity=normal 4    <conf:(1)> lift:(2) lev:(0.14) [2] conv:(2)
 3. humidity=normal windy=FALSE 4 ==> play=yes 4    <conf:(1)> lift:(1.56) lev:(0.1) [1] conv:(1.43)
 4. outlook=sunny play=no 3 ==> humidity=high 3    <conf:(1)> lift:(2) lev:(0.11) [1] conv:(1.5)
 5. outlook=sunny humidity=high 3 ==> play=no 3    <conf:(1)> lift:(2.8) lev:(0.14) [1] conv:(1.93)
 6. outlook=rainy play=yes 3 ==> windy=FALSE 3    <conf:(1)> lift:(1.75) lev:(0.09) [1] conv:(1.29)
 7. outlook=rainy windy=FALSE 3 ==> play=yes 3    <conf:(1)> lift:(1.56) lev:(0.08) [1] conv:(1.07)
 8. temperature=cool play=yes 3 ==> humidity=normal 3    <conf:(1)> lift:(2) lev:(0.11) [1] conv:(1.5)
 9. outlook=sunny temperature=hot 2 ==> humidity=high 2    <conf:(1)> lift:(2) lev:(0.07) [1] conv:(1)
10. temperature=hot play=no 2 ==> outlook=sunny 2    <conf:(1)> lift:(2.8) lev:(0.09) [1] conv:(1.29)
*/
