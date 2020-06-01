package irisDataExamples;

/**
 * Filter data with WEKA. 
 * 
 * Remove 2nd and 3rd attributes on Iris data and saves in "src/data/iris_afterAttributesFilter.arff".
 * 
 * @author feBueno, May 2020
 * fernando.bueno.gutie@gmail.com
 */

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import java.io.File;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class AttributesFilter {
	public static void main(String[] args) throws Exception {
		
		//load dataset
		DataSource source = new DataSource("src/data/iris.arff");
		Instances iris_ds = source.getDataSet();
		
		//remove 2nd and 3rd attributes
		//weka.filters.unsupervised.attribute.Remove
		//String[] opts = new String[] {"-R","2,3"};//approach1
		String[] opts = new String[] {"-R","1,4","-V"};//approach2

		//create a remove object (filter class)
		Remove remove = new Remove();
		//Set filter options
		remove.setOptions(opts);
		//pass dataset to filter
		remove.setInputFormat(iris_ds);
		//apply filter
		Instances iris_afterRemove_ds = Filter.useFilter(iris_ds,  remove);
		
		
		//save dataset
		ArffSaver saver = new ArffSaver();
		saver.setInstances(iris_afterRemove_ds);
		saver.setFile(new File("src/data/iris_afterAttributesFilter.arff"));
		saver.writeBatch();
		
		System.out.println("Filtered file was saved in src/data/iris_afterAttributesFilter.arff");
	}
} /*Output: 
	Filtered file was saved in src/data/iris_afterAttributesFilter.arff
	*/
