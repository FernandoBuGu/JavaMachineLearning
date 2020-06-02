package irisDataExamples;

/**
 * Regression estimates using WEKA. 
 * 
 * Linear Regression, Support Vector Machine and Multilayer Perceptron 
 * are applied on the Iris data, and the coefficient estimates are printed.
 * 
 * REQUIRE:
 * Depending on Weka installation, it may be required to add arpack_combined.jar core.jar and mtj.jar from
 * https://svn.cms.waikato.ac.nz/svn/weka/branches/stable-3-8/weka/lib/
 * 
 * @author feBueno, May 2020
 * fernando.bueno.gutie@gmail.com
 */

import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMOreg;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;


public class RegressionCoefficients {
	public static void main(String[] args) throws Exception {
		
		//load dataset
		DataSource source = new DataSource("src/data/iris.arff");
		Instances data = source.getDataSet();
	
		//set class index to the 2nd last attribute (quantitative)
		data.setClassIndex(data.numAttributes()-2);
		
		//build model
		LinearRegression lr = new LinearRegression();
		lr.buildClassifier(data);
		System.out.println("=====linear regression=====");
		System.out.println(lr);
		
		SMOreg SMOr = new SMOreg();
		SMOr.buildClassifier(data);
		System.out.println("=====support vector machine regression=====");
		System.out.println(SMOr);
		
		MultilayerPerceptron MLP = new MultilayerPerceptron();
		MLP.buildClassifier(data);
		System.out.println("=====multilayer perceptron=====");
		System.out.println(MLP);
	}
} /*Output:
=====linear regression=====

Linear Regression Model

petalwidth =

     -0.0948 * sepallength +
      0.2497 * sepalwidth +
      0.2409 * petallength +
      0.6582 * class=Iris-versicolor,Iris-virginica +
      0.3995 * class=Iris-virginica +
     -0.4878
=====support vector machine regression=====
SMOreg

weights (not support vectors):
 -       0.0767 * (normalized) sepallength
 +       0.1983 * (normalized) sepalwidth
 +       0.6097 * (normalized) petallength
 -       0.2101 * (normalized) class=Iris-setosa
 +       0.0236 * (normalized) class=Iris-versicolor
 +       0.1865 * (normalized) class=Iris-virginica
 +       0.1156



Number of kernel evaluations: 11325 (92.805% cached)
=====multilayer perceptron=====
Linear Node 0
    Inputs    Weights
    Threshold    0.35263393308458785
    Node 1    -1.3177638726899452
    Node 2    -1.3973907171424629
    Node 3    0.6361087962994509
Sigmoid Node 1
    Inputs    Weights
    Threshold    -2.6238993873712517
    Attrib sepallength    2.1979220771672248
    Attrib sepalwidth    2.2006846543671386
    Attrib petallength    0.08940182722130202
    Attrib class=Iris-setosa    1.0070179203449823
    Attrib class=Iris-versicolor    0.8167703626226193
    Attrib class=Iris-virginica    0.80820050595105
Sigmoid Node 2
    Inputs    Weights
    Threshold    -1.014115454603922
    Attrib sepallength    0.1158422425059921
    Attrib sepalwidth    -0.6800894013664917
    Attrib petallength    -1.822010479943909
    Attrib class=Iris-setosa    1.2838144400321096
    Attrib class=Iris-versicolor    0.386148348445481
    Attrib class=Iris-virginica    -0.6407831387506644
Sigmoid Node 3
    Inputs    Weights
    Threshold    -1.7528421350514183
    Attrib sepallength    0.708677846803865
    Attrib sepalwidth    3.532972470597868
    Attrib petallength    2.094233477394254
    Attrib class=Iris-setosa    -0.010144194007702728
    Attrib class=Iris-versicolor    0.6714200127203287
    Attrib class=Iris-virginica    1.1684952905452755
Class 
    Input
    Node 0
*/
