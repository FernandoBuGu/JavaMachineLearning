package irisDataExamples;

/**
 * Apply classifiers on filtered data using WEKA
 * 
 * A J48 classifier is applied on the Iris data after removing the 2nd variable.
 * Then, a graph is printed
 * 
 * @author feBueno, May 2020
 * fernando.bueno.gutie@gmail.com
 */

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import java.io.File;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;


public class FilterClassifier {
	public static void main(String[] args) throws Exception {
		
		//load dataset
		DataSource source = new DataSource("src/data/iris.arff");
		Instances data = source.getDataSet();
		
		//set class index to the last attribute (categorical)
		data.setClassIndex(data.numAttributes()-1);
		
		J48 tree = new J48();
		Remove remove = new Remove();

		String[] classifierOptions = new String[4];
		classifierOptions[0] = "-C"; classifierOptions[1]="0.11";//confidence threshold for pruning.
		classifierOptions[2] = "-M"; classifierOptions[3]="4";//minimum number of instances per leaf
		tree.setOptions(classifierOptions);
		
		String[] filterOptions = new String[] {"-R","2"};
		remove.setOptions(filterOptions);
		
		//weka.classifiers.meta.FilteredClassifier
		FilteredClassifier fc = new FilteredClassifier();//setInputFormat not required
		fc.setFilter(remove);
		fc.setClassifier(tree);
		fc.buildClassifier(data);

		
		System.out.println(tree.graph());
		
	}
} /*Output:
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
