package basics;

/**
 * Example of linear regression on the iris dataset using WEKA 
 * 
 * REQUIRE:
 * Depending on Weka installation, it may be required to add arpack_combined.jar core.jar and mtj.jar from
 * https://svn.cms.waikato.ac.nz/svn/weka/branches/stable-3-8/weka/lib/
 * 
 * @author feBueno, May 2020
 * fernando.bueno.gutie@gmail.com
 */

import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.SMOreg;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;


public class RegressionExample {
	public static void main(String[] args) throws Exception {
		
		//load dataset
		DataSource source = new DataSource("src/data/iris.arff");
		Instances data = source.getDataSet();
	
		//set class index to the 2nd last attribute (quantitative)
		data.setClassIndex(data.numAttributes()-2);
		
		//build model
		LinearRegression lr = new LinearRegression();
		lr.buildClassifier(data);
		System.out.println(lr);
		
		SMOreg SMOr = new SMOreg();
		SMOr.buildClassifier(data);
		System.out.println(SMOr);
		
	}
}
