package irisDataExamples;

/**
 * Converts data to Weka-sparse format.
 * 
 * Converted data is saved in src/data/iris_afterSparseHandle.arff
 * 
 * Example of weka-sparse format:
 * 	NonSparse: 1,1,0,1,0,0,0,0,0,1,3.2
 * 	Sparse: 0 1, 1 1, 3 1, 9 1, 10 3.2
 * 	where in each comma-separated-element the first number is the index, the second is the value, and the 0-values are ignored
 * 
 * 
 * @author feBueno, May 2020
 * fernando.bueno.gutie@gmail.com
 */

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import java.io.File;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.NonSparseToSparse;


public class SparsityHandler {
	public static void main(String[] args) throws Exception {
	
		//load dataset
		DataSource source = new DataSource("src/data/iris.arff");
		Instances iris_ds = source.getDataSet();
		
		//create NonSparseToSparse object
		weka.filters.unsupervised.instance.NonSparseToSparse sp = new NonSparseToSparse();
		
		//apply to dataset
		sp.setInputFormat(iris_ds);
		Instances iris_afterFilter_ds = Filter.useFilter(iris_ds, sp);
		
		//save
		ArffSaver saver = new ArffSaver();
		saver.setInstances(iris_afterFilter_ds);
		saver.setFile(new File("src/data/iris_afterSparseHandle.arff"));
		saver.writeBatch();
		
		System.out.println("Converted data was saved in src/data/iris_afterSparseHandle.arff");

	}
} /*Output: 
Converted data was saved in src/data/iris_afterSparseHandle.arff
*/
