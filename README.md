# EM-for-GMM
Implementation of EM fitting of a mixture of gaussians on the two-dimensional data set

Author: Md Kamrul Hasan
Date: 31th March, 2017

===============================================================================================

Implementation of EM fitting of a mixture of gaussians on the two-dimensional data set
I had tried different numbers of mixtures, as well as tied vs. separate covariance matrices 
for each gaussian.

Run instruction: python gmm_em.py (make sure points.dat in the same directory)


===============================================================================================

Init EM:
	I have randomly chose k (num of cluster) data point to initilize k means.
	And also intitlize k covariance to make sure determinate is non zero value 


Output:

Five files:

  1.seperate_cov_training.png :
  	 log likelihood on train  vs iteration for different numbers of mixtures. I have used separate 
     covariance matrices for each gaussian.

  2.seperate_cov_dev.png:
  	 log likelihood on train  vs iteration for different numbers of mixtures. I have used separate 
     covariance matrices for each gaussian.


  3. tied_cov_training.png:
      log likelihood on traing  vs iteration for different numbers of mixtures. I have used tied 
      covariance matrices for each gaussian.

  4. tied_cov_dev.png:
  	 log likelihood on dev  vs iteration for different numbers of mixtures. I have used tied covariance 
     matrices for each gaussian.

  5. scatter.png:
     scatter plot for all data


===============================================================================================

 Result Analysis:
 From scatter plot it can be guessed that the number of cluster should vary among [4,5,6,7]. From 
 log_likelihood graph we can determine the number of appropritae cluster cluster. From both training
  and dev data loglikehood graph, for which k the graph show highest log likehood with less fluctuations 
  is good choice for number of clusters. For here , k=5,6 or 7 is almost similar. So, they are the best 
  choice for clustering. But as it is random algorithm, so it can vary. But I think good choice will  
  be either 5, 6 or 7 This conclusion also make sense if we see the scatter graph.

 For tied covariance, it convergences very quickly compare to the seperate covariances. 
