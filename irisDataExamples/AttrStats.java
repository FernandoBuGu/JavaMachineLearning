package irisDataExamples;

/**
 * Print attributes' stats with WEKA
 *
 * On iris data, get distinct, unique, missing, mean... values for each attribute
 * Manipulate instances (i.e. compute z-scores...)
 * 
 * @author feBueno, May 2020
 * fernando.bueno.gutie@gmail.com
 */

import java.io.File;
import java.text.DecimalFormat;
import java.util.Arrays;

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
	
		
		System.out.println("number of attributes: " + data.numAttributes());

		//set class index as the 2nd last attribute (petalwidth)
		if(data.classIndex()==-1)
			data.setClassIndex(data.numAttributes()-2);

		//specify format for printing numerics
		DecimalFormat f = new DecimalFormat("##.00");
		
		//get number of attributes
		int numAttr = data.numAttributes();
		for(int i=0;i<numAttr;i++) {//attributes are 0-based
			
			//get an attributeStats object
			AttributeStats as = data.attributeStats(i);//weka.core.AttributeStats
			
			//print distinct counts
			int dc = as.distinctCount;//approach 1 to get distinct values. 
			//as.uniqueCount would return number of values that only appear once
			System.out.println("The "+i+" attribute has "+dc+" distinct values");
		
			//print mean if numeric
			if(data.attribute(i).isNumeric()) {
				System.out.println("The "+i+" attribute is numeric");
				Stats s = as.numericStats;
				System.out.println("The "+i+" attribute mean(sd) value is "+ f.format(s.mean) + "(" + f.format(s.stdDev) +")");
			}
			
			//check if attr is nominal
			if(data.attribute(i).isNominal()) {
				System.out.println("The "+i+" attribute is nominal");
				int n = data.attribute(i).numValues();//approach 1 to get distinct values
				System.out.println("The "+i+" attribute has "+n+" categories");
				int[] nomCounts_intArr = as.nominalCounts;//get values from each class 
				System.out.println("The "+i+" attribute has "+Arrays.toString(nomCounts_intArr)+" elements in each of the categories");

			}
		}
		
		//print attibute name where index is set
		System.out.println("Index is at: " + data.attribute(data.classIndex()).name());
		
		//get number of instances (rows)
		int numInst = data.numInstances();
		//loop through the first 6 instances;
		System.out.println("The first 6 instances are printed with their z-score for the index attribute:");
		for(int j=0;j<6;j++) {
			Instance minstance = data.instance(j);

			if(minstance.classIsMissing()) {
				System.out.println("The class is missing in the "+j+" attribute");
			}
			double cV= minstance.classValue();
			System.out.print(minstance + " : ");//print the whole instance
			
			//print value of the instance in the attribute selected as classIndex
			System.out.print(data.attribute(data.classIndex()).name()+": ");//att name
			
			//get statistics from index class for each instance in loop
			AttributeStats as = data.attributeStats(data.classIndex());//weka.core.AttributeStats
			Stats s = as.numericStats;

			if(data.attribute(data.classIndex()).isNominal()) {
				System.out.println(minstance.classAttribute().value((int)cV));
			} else {
				System.out.print(cV+ " : ");
				System.out.print(data.attribute(data.classIndex()).name()+" z-score: ");//att name
				System.out.println(f.format(cV-(double)s.mean/(double)s.stdDev));
			}
		}
		
	}
} /*Output:
number of attributes: 5
The 0 attribute has 35 distinct values
The 0 attribute is numeric
The 0 attribute mean(sd) value is 5.84(.83)
The 1 attribute has 23 distinct values
The 1 attribute is numeric
The 1 attribute mean(sd) value is 3.05(.43)
The 2 attribute has 43 distinct values
The 2 attribute is numeric
The 2 attribute mean(sd) value is 3.76(1.76)
The 3 attribute has 22 distinct values
The 3 attribute is numeric
The 3 attribute mean(sd) value is 1.20(.76)
The 4 attribute has 3 distinct values
The 4 attribute is nominal
The 4 attribute has 3 categories
The 4 attribute has [50, 50, 50] elements in each of the categories
Index is at: petalwidth
The first 6 instances are printed with their z-score for the index attribute:
5.1,3.5,1.4,0.2,Iris-setosa : petalwidth: 0.2 : petalwidth z-score: -1.37
4.9,3,1.4,0.2,Iris-setosa : petalwidth: 0.2 : petalwidth z-score: -1.37
4.7,3.2,1.3,0.2,Iris-setosa : petalwidth: 0.2 : petalwidth z-score: -1.37
4.6,3.1,1.5,0.2,Iris-setosa : petalwidth: 0.2 : petalwidth z-score: -1.37
5,3.6,1.4,0.2,Iris-setosa : petalwidth: 0.2 : petalwidth z-score: -1.37
5.4,3.9,1.7,0.4,Iris-setosa : petalwidth: 0.4 : petalwidth z-score: -1.17
*/
