package irisDataExamples;

/**
 * Discretize data variables in Weka
 * 
 * The 3rd and 4th variables from Iris are discretized to create 4 bins
 * Discretized data is saved in src/data/iris_afterDiscretizeAttributes.arff
 * 
 * @author feBueno, May 2020
 * fernando.bueno.gutie@gmail.com
 */

import java.io.File;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

public class DiscretizeAttributes {
	public static void main(String[] args) throws Exception {

		//load dataset
		DataSource source = new DataSource("src/data/iris.arff");
		Instances iris_ds = source.getDataSet();
		
		//set options
		String[] options =new String[5];
		//chose the number of intervals
		options[0] = "-B"; options[1] = "4";//number of bins 
		options[2]="-R"; options[3]="1-2";
		options[4]="-V";//The 3rd and 4th attribute will be discretize
		
		Discretize discretize = new Discretize();
		discretize.setOptions(options);
		discretize.setInputFormat(iris_ds);
		Instances iris_afterDiscretizeAttributes_ds = Filter.useFilter(iris_ds, discretize);
		
		//save
		ArffSaver saver = new ArffSaver();
		saver.setInstances(iris_afterDiscretizeAttributes_ds);
		saver.setFile(new File("src/data/iris_afterDiscretizeAttributes.arff"));
		saver.writeBatch();
		
		System.out.println("Discretized data was saved in src/data/iris_afterDiscretizeAttributes.arff");
	}
} /*Output:
Discretized data was saved in src/data/iris_afterDiscretizeAttributes.arff
*/
