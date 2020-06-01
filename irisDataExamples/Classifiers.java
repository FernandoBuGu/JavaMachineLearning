package irisDataExamples;

/**
 * Print classifiers information given data, using WEKA
 * 
 * On Iris data, print capabilities of NaiveBayes, sequential minimal optimization and J48;
 * estimates class probabilities for 10th instance for one of the instances; 
 * and a graph constructed with the J48 
 * 
 * @author feBueno, May 2020
 * fernando.bueno.gutie@gmail.com
 */

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import java.io.File;
import java.util.Arrays;

import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class Classifiers {
	public static void main(String[] args) throws Exception {
		
		//load dataset
		DataSource source = new DataSource("src/data/iris.arff");
		Instances data = source.getDataSet();
		
		//set class index to the last attribute (categorical)
		data.setClassIndex(data.numAttributes()-1);
		
		//NaiveBayes
		System.out.println("=====NaiveBayes=====");
		NaiveBayes nb = new NaiveBayes();
		nb.buildClassifier(data);
		System.out.println(nb.getCapabilities().toString());
		System.out.println("Estimates class probabilities for 10th instance: "+Arrays.toString(nb.distributionForInstance(data.instance(10))));
		
		//Sequential Minimal Optimization
		System.out.println("=====Sequential Minimal Optimization=====");
		SMO svm = new SMO();//sequential minimal optimization
		svm.buildClassifier(data);
		System.out.println(svm.getCapabilities().toString());
		svm.setNumFolds(10);
		System.out.println("Estimates class probabilities for 10th instance: "+Arrays.toString(svm.distributionForInstance(data.instance(10))));

		//J48
		System.out.println("=====J48=====");
		String[] options = new String[4];
		options[0] = "-C"; options[1]="0.11";//confidence threshold for pruning
		options[2] = "-M"; options[3]="4";//minimum number of instances per leaf
		J48 tree = new J48();
		tree.setOptions(options);
		tree.buildClassifier(data);
		System.out.println("Estimates class probabilities for 10th instance: "+Arrays.toString(tree.distributionForInstance(data.instance(10))));
		System.out.println(tree.graph());
		
	}
} /*Output:
=====NaiveBayes=====
Jun 01, 2020 7:31:27 AM com.github.fommil.netlib.ARPACK <clinit>
WARNING: Failed to load implementation from: com.github.fommil.netlib.NativeSystemARPACK
Jun 01, 2020 7:31:27 AM com.github.fommil.netlib.ARPACK <clinit>
WARNING: Failed to load implementation from: com.github.fommil.netlib.NativeRefARPACK
Capabilities: [Nominal attributes, Binary attributes, Unary attributes, Empty nominal attributes, Numeric attributes, Missing values, Nominal class, Binary class, Missing class values]
Dependencies: []
Interfaces: [WeightedAttributesHandler, WeightedInstancesHandler]
Minimum number of instances: 0

Estimates class probabilities for 10th instance: [1.0, 4.9134129544166216E-17, 1.5180817919099027E-24]
=====Sequential Minimal Optimization=====
Capabilities: [Nominal attributes, Binary attributes, Unary attributes, Empty nominal attributes, Numeric attributes, Missing values, Nominal class, Binary class, Missing class values]
Dependencies: [Nominal attributes, Binary attributes, Unary attributes, Empty nominal attributes, Numeric attributes, Date attributes, String attributes, Relational attributes]
Interfaces: [WeightedInstancesHandler]
Minimum number of instances: 1

Estimates class probabilities for 10th instance: [0.6666666666666666, 0.3333333333333333, 0.0]
=====J48=====
Estimates class probabilities for 10th instance: [1.0, 0.0, 0.0]
digraph J48Tree {
N0 [label="petalwidth" ]
N0->N1 [label="<= 0.6"]
N1 [label="Iris-setosa (50.0)" shape=box style=filled ]
N0->N2 [label="> 0.6"]
N2 [label="petalwidth" ]
N2->N3 [label="<= 1.7"]
N3 [label="petallength" ]
N3->N4 [label="<= 4.9"]
N4 [label="Iris-versicolor (48.0/1.0)" shape=box style=filled ]
N3->N5 [label="> 4.9"]
N5 [label="Iris-virginica (6.0/2.0)" shape=box style=filled ]
N2->N6 [label="> 1.7"]
N6 [label="Iris-virginica (46.0/1.0)" shape=box style=filled ]
}
*/
