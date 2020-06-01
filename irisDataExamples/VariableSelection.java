package irisDataExamples;

/**
 * Attribute selection in Weka.
 * 
 * Variable selection is carried on the Iris based on 
 * individual performance and redundancy with other variables.
 * A greedy stepwise algorithm is used.
 * This leads to removal of Sepal.length and Sepal.width.
 * After attributes removal, data is saved in src/data/iris_afterMyAttributeSelection.arff
 *
 * 
 * @author feBueno, May 2020
 * fernando.bueno.gutie@gmail.com
 */

import java.io.File;

import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;

public class VariableSelection {
	public static void main(String[] args) throws Exception {

		//load dataset
		DataSource source = new DataSource("src/data/iris.arff");
		Instances iris_ds = source.getDataSet();
	
		//attribute selection
		//create AttributeSelection object
		AttributeSelection filter = new AttributeSelection();
		//create evaluator
		CfsSubsetEval eval = new CfsSubsetEval();//feature selection based on individual performance and redundancy with other variables
		//searh algorithm object
		GreedyStepwise search = new GreedyStepwise();//adds/deletes features iteratively and stops when performance increases no more
		//set backwards search
		search.setSearchBackwards(true);
		//set the filter to use the evaluator and search algorithm
		filter.setEvaluator(eval);
		filter.setSearch(search);
		//apply
		filter.setInputFormat(iris_ds);
		Instances iris_afterMyAttributeSelection_ds = Filter.useFilter(iris_ds,  filter);
		
		//save
		ArffSaver saver = new ArffSaver();
		saver.setInstances(iris_afterMyAttributeSelection_ds);
		saver.setFile(new File("src/data/iris_afterMyAttributeSelection.arff"));
		saver.writeBatch();
		
		System.out.println("Filtered file was saved in src/data/iris_afterMyAttributeSelection.arff");
	}
} /*Output: 
Filtered file was saved in src/data/iris_afterMyAttributeSelection.arff
*/
