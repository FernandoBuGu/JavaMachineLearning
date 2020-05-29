package basics;

/**
 * Attribute selection in weka
 * CfsSubsetEval with GreedyStepwise is used on the Iris data and Sepal.length is removed 
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

public class MyAttributeSelection {
	public static void main(String[] args) throws Exception {

		//load dataset
		DataSource source = new DataSource("src/data/iris.arff");
		Instances iris_ds = source.getDataSet();
	
		//attribute selection
		//create AttributeSelection object
		AttributeSelection filter = new AttributeSelection();
		//create evaluator
		CfsSubsetEval eval = new CfsSubsetEval();//feature selection based on individual perfomance and redundance with other variables
		//searh algorithm object
		GreedyStepwise search = new GreedyStepwise();//adds/deletes features iteratively and stops when perfomance increases no more
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
	}
}
