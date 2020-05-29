package basics;

/**
 * Apply classifiers on filtered data using WEKA
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
		
		String[] filterOptions = new String[] {"-R","-1"};
		remove.setOptions(filterOptions);
		
		//weka.classifiers.meta.FilteredClassifier
		FilteredClassifier fc = new FilteredClassifier();//setInputFormat not required
		fc.setFilter(remove);
		fc.setClassifier(tree);
		fc.buildClassifier(data);

		
		System.out.println(tree.graph());
		
	}
}
