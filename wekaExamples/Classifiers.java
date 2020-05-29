package basics;

/**
 * Classifiers using WEKA
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
		
		NaiveBayes nb = new NaiveBayes();
		nb.buildClassifier(data);
		System.out.println(nb.getCapabilities().toString());
		
		SMO svm = new SMO();//sequential minimal optimization))
		svm.buildClassifier(data);
		System.out.println(svm.getCapabilities().toString());

		String[] options = new String[4];
		options[0] = "-C"; options[1]="0.11";//confidence threshold for pruning.
		options[2] = "-M"; options[3]="4";//minimum number of instances per leaf

		J48 tree = new J48();
		tree.setOptions(options);
		tree.buildClassifier(data);
		System.out.println(tree.graph());
		
		//save dataset
		/*ArffSaver saver = new ArffSaver();
		saver.setInstances(iris_afterRemove_ds);
		saver.setFile(new File("src/data/iris_afterAttributesFilter.arff"));
		saver.writeBatch();*/
		
	}
}
