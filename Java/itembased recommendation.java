package testforitem.testitem;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.ItemBasedRecommender;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args ) throws Exception
    {
    	DataModel model = new FileDataModel(new File("dataforitem/instantvideo.csv"));
    	ItemSimilarity similarity = new PearsonCorrelationSimilarity(model);
    	ItemBasedRecommender recommender = new GenericItemBasedRecommender(model, similarity);
    	List<RecommendedItem> recommendations = recommender.recommend(26261, 10);
    	for (RecommendedItem recommendation : recommendations) {
    	  System.out.println(recommendation);
    	}
    }
}
