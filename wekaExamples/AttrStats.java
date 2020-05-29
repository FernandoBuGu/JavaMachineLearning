package basics;

/**
 * Attribute stats in the iris dataset using WEKA
 *
 * @author feBueno, May 2020
 * fernando.bueno.gutie@gmail.com
 */

import java.io.File;

import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.experiment.Stats;

public class AttrStats {
	public static void main(String[] args) throws Exception {

		//load dataset
		DataSource source = new DataSource("src/data/iris.arff");
		Instances data = source.getDataSet();
	
		//set class index as the last attribute
		if(data.classIndex()==-1)
			data.setClassIndex(data.numAttributes()-1);
		
		//get number of attributes
		int numAttr = data.numAttributes()-1;
		for(int i=0;i<numAttr;i++) {
			//check if attr is not nominal
			if(data.attribute(i).isNominal()) {
				System.out.println("The "+i+" attribute is nominal");
				int n = data.attribute(i).numValues();
				System.out.println("The "+i+" attribute has "+n+" values");
			}
			//get an attributeStats object
			AttributeStats as = data.attributeStats(i);
			int dc = as.distinctCount;
			System.out.println("The "+i+" attribute has "+dc+" distinct values");
		
			if(data.attribute(i).isNumeric()) {
				System.out.println("The "+i+" attribute is numeric");
				Stats s = as.numericStats;
				System.out.println("The "+i+" attribute mean value is "+ s.mean);
			}
		}
		//get number of instances
		int numInst = data.numInstances();
		//loop through all instances;
		for(int j=0;j<numInst;j++) {
			Instance minstance = data.instance(j);

			if(minstance.classIsMissing()) {
				System.out.println("The class is missing in the "+j+" attribute");
			}
			double cV= minstance.classValue();
			System.out.println(minstance.classAttribute().value((int)cV));
		}
		
	}
}
