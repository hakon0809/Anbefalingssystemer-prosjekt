from sklearn.metrics import mean_squared_error
import numpy as np

def evaluate_recall(pred, actual):
    """
    Evaluate recommendations according to recall

    input:
      pred: 	an array of predicted recommendations, where each item is an array of article IDs.
							the index corresponds to an event, in the same order as the "actual" variable,
							while the item with that index is a list of predicted articles in priority order
							which our system has generated for the userId and time of the original event
							for example, our prediction for the 51st event, pred[50] might be [107, 108, 95, 115, 120],
							where 107 is the article id our system most expects that particular user clicked on next
							the example code and example report uses a "k" value.
							the k value is simply how many articles we are allowed to predict.
      actual: an array of actual best recommendations.
							the index corresponds to an event, in the same order as the "pred" variable,
							while the item with that index is an integer, the articleId the user of that event actually clicked on.
							for example, for the 51st event, actual[50] might be 95
							meaning the user in that event clicked on the article with id 95.
							if the articleId in actual[50] is the same as one of the articleIds in pred[50],
							that means we successfully recommended the article the user clicked on (as one of many)
    output:
      a scalar value, the recall metric: the proportion of true positives compared to all evaluated events
			for example, an output of 0.5 would indicate that for half of the events we evaluate:
			our system outputted a list of (k) articleIds, pred[i],
			and one of those articles was the actual article that got clicked on, actual[i].
    """
    total_num = len(actual) # number of evaluated events
    tp = 0. # true positives, where the actually clicked articleId is in our recommendations

    # Iterate over predicted and actual results together (pred[i] = p, actual[i] = a, for each i)
    for p, a in zip(pred, actual):
        if a in p:
            # If the article that got clicked on in this event was in the recommendations
            # that counts as a true positive (the user decided to click one of the articles we wanted to recommend)
            tp += 1.

    # Divide the number of successes by the total number of events
    # This gives us our recall metric
    recall = tp / float(total_num)
    return recall

def evaluate_arhr(pred, actual):
    """
    Evaluate recommendations according to Average Reciprocal Hit-Rate

    input:
      pred: 	an array of predicted recommendations, where each item is an array of article IDs
							sorted most to least relevant/confident.
							the index corresponds to an event, in the same order as the "actual" variable,
							while the item with that index is a list of predicted articles in priority order
							which our system has generated for the userId and time of the original event
							for example, our prediction for the 51st event, pred[50] might be [107, 108, 95, 115, 120],
							where 107 is the article id our system most expects that particular user clicked on next
							the example code and example report uses a "k" value.
							the k value is simply how many articles we are allowed to predict.
      actual: an array of actual best recommendations.
							the index corresponds to an event, in the same order as the "pred" variable,
							while the item with that index is an integer, the articleId the user of that event actually clicked on.
							for example, for the 51st event, actual[50] might be 95
							meaning the user in that event clicked on the article with id 95.
							if the articleId in actual[50] is the same as one of the articleIds in pred[50],
							that means we successfully recommended the article the user clicked on (as one of many)
							for ARHR, that score is scaled by how far down our recommendations that is.
							in this case, we scored 0.33 on even 50. (1/3, because it was the 3rd recommendation on our list)
    output:
      a scalar value, the ARHR score for the given data.
			this is the average score of all evaluated events.
			it is Averaged across all scores.
			it is Reciprocal, meaning it is inversely proportional to the articles position in our recommendations.
			for example, an output of 0.5 could indicate multiple things:
			we could recommend the right article as number 1 half the time, and fail to recommend it the other half
			we could recommend the right article as number 2 100% of the time
			anything in between these two scenarios
    """
    trhr = 0. # Total Reciprocal Hit-Rate, the sum of the individual scores of all hits so far

    # Iterate over predicted and actual results together
    for p, a in zip(pred, actual):
        if a in p:
            # For each hit (when the article is in the predictions), increase the score sum by
            # 1 / the rank of the recommendation (+1 to convert from 0-index to 1-index)
            # We only add to the score on hits. If the article is not in the recommendations, we "add" 0 to our score
            trhr += 1./float(p.index(a) + 1.)

    # Divide the sum of our individual scores by the number of events we evaluated, to get the Average Reciprocal Hit-Rate
    arhr = trhr / len(actual)
    return arhr

def evaluate_mse(pred, actual):
    """
    Evaluate predicted scores according to Mean Squared Error
		(not the same input format as the other two.
		this one expects predicted scores, not a list of recommendations.)

    input:
      pred: 	an array of predicted scores, where each item is a score for one user and one article.
							the example code uses this on a flattened matrix segment.
							in the unflattened matrix, the first coordinate was userid, and the second was articleid,
							with the value being the rating.
							this corresponds to the predicted values for the entire "test" section of the dataset
      actual: an array of actual scores, where each item is a score from one user given to one article.
							this is the same format as "pred", 
							except the values are generated from the dataset, instead of by an algorithm
							this corresponds to the actual values for the entire "test" section of the dataset
							if you have three users and two articles in the test set, this array would have 6 entries.
							either as a 3x2 matrix, or as a flattened 1x6 matrix 
    output:
      a scalar value, the mean squared error score for the given data.
			
			the example code only checks for the error on the user-article pairs where
			the user actually opened the article (and thus, gave a nonzero rating)
			so it ignores predicted ratings for articles that didnt get seen,
			but this code does not do that automatically.
			
			for example, if the provided ratings are binary (0 for not clicked, 1 for clicked),
			an output of 1 would imply that we predicted the users would click none of the articles they clicked (100% wrong)
			(but if we have removed nonzero ratings, we could still have correctly identified articles the users didnt click on)
			
			for example, if the provided ratings are positive seconds (score of 60 means they spent 1 minute on the article),
			an output of 100 would imply that we were, on average, around 10 seconds off on our predictions,
			
    """
    return mean_squared_error(actual, pred)