import numpy as np 
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
import random

#initilization of means, covariances and mixing coefficients
#make sure the determinants

def init_em_params(data,num_clusters):
	num_dim=len(data[0])
	means=[]
	for i in range(num_clusters):
		index=random.randint(0,len(data)-1)
		means.append(data[index])
	means=np.array(means)

	covs=[]
	for k in range(num_clusters):
		cov=np.zeros(shape=(num_dim,num_dim))
		for j in range(num_dim):
			cov[j][j]=float(random.randint(1,80))/80
		covs.append(cov)

	mixing_coeffs=np.random.random(num_clusters)
	mixing_coeffs /= mixing_coeffs.sum()

	return means,covs,mixing_coeffs



def log_likelihood(data,means,covs,mixing_coeffs):
	ll=0
	num_clusters = len(mixing_coeffs)
	for i in range(len(data)):
		sum_resp=0
		for k in range(num_clusters):
			p=multivariate_normal.pdf([data[i]], mean=means[k],cov=covs[k]);
			sum_resp+=mixing_coeffs[k]*p

		ll+=np.log(sum_resp)
	return ll

def update_means(data,resp,N_K,num_data,num_clusters):
	num_dim=len(data[0])
	means = [np.zeros(len(data[0]))] * num_clusters
	for k in range(num_clusters):
		sum_x=np.zeros(num_dim)
		for i in range(num_data):
			sum_x+=resp[i,k]*data[i]
		means[k]=sum_x/N_K[k]
	return means


def update_covariances(data,means,resp,N_K,num_data,num_clusters):
	num_dim=len(data[0])
	covs= [np.zeros((num_dim,num_dim))] * num_clusters
	for k in range(num_clusters):
		sum_k=np.zeros((num_dim,num_dim))
		for i in range(num_data):
			x=data[i]-means[k]
			x=resp[i,k]*np.outer(x,x)
			sum_k+=x
		covs[k]=sum_k/N_K[k]
	#this commented portion is for tied covariances
	# sum_cov=np.sum(covs, axis=0)/num_clusters
	# tied_covs=[sum_cov]*num_clusters
	# return tied_covs

	return covs

def update_mixing_coefficient(N_K,num_data,num_clusters):
	mixing_coeffs=np.zeros(num_clusters)
	for k in range(num_clusters):
		mixing_coeffs[k]=N_K[k]/num_data
	return mixing_coeffs

def e_step(data,means,covs,mixing_coeffs):
	num_data = len(data)
	num_clusters = len(mixing_coeffs)
	resp = np.zeros((num_data, num_clusters))	
	for i in range(num_data):
	    for k in range(num_clusters):
	    	p=multivariate_normal.pdf([data[i]], mean=means[k],cov=covs[k]);
	    	resp[i,k]=mixing_coeffs[k]*p

	row_sums = resp.sum(axis=1)[:, np.newaxis]
	resp = resp / row_sums	
	return resp

def m_step(data,resp):
	num_clusters=len(resp[0])
	num_data=len(data)
	N_K=np.sum(resp, axis=0)
	means=update_means(data,resp,N_K,num_data,num_clusters)
	covs=update_covariances(data,means,resp,N_K,num_data,num_clusters)
	mixing_coeffs=update_mixing_coefficient(N_K,num_data,num_clusters)

	return means,covs,mixing_coeffs



def main():
	data=np.loadtxt("points.dat")
	training_data=data[0:899]
	dev_data=data[900:999]
	maxiter=100
	thresh=1e-6
	K=list(range(2,3))

	all_t_ll=[]
	all_d_ll=[]
	#run for different clusters
	for num_clusters in K:

		means,covs,mixing_coeffs=init_em_params(training_data,num_clusters)
		training_ll=[]
		dev_ll=[]

		prev_t_ll=-100000000

		for itr in range(maxiter):

			resp=e_step(training_data,means,covs,mixing_coeffs)

			means,covs,mixing_coeffs=m_step(training_data,resp)		

			t_ll=log_likelihood(training_data,means,covs,mixing_coeffs)
			training_ll.append(t_ll)

			d_ll=log_likelihood(dev_data,means,covs,mixing_coeffs)
			dev_ll.append(d_ll)
			# if the cnahge is less than threshhold stopiteration (converges)
			delta=np.absolute(t_ll - prev_t_ll)
			if  delta < thresh and t_ll > -np.inf:
				break

			prev_t_ll=t_ll

		all_t_ll.append(training_ll)
		all_d_ll.append(dev_ll)

	legend=[]
	for i in range(len(all_t_ll)):
		plt.plot(all_t_ll[i],linestyle='--', marker='.')
		legend.append('k='+str(i+2))

	plt.ylabel('log likelihood in training data (diff num of mixtures)')
	plt.xlabel('iteration')
	plt.legend(legend, loc='lower right')
	plt.show()

	legend=[]
	for i in range(len(all_d_ll)):
		plt.plot(all_d_ll[i],linestyle='--', marker='.')
		legend.append('k='+str(i+2))

	plt.ylabel('log likelihood in dev data (diff num of mixtures)')
	plt.xlabel('iteration')
	plt.legend(legend, loc='lower right')
	plt.show()



if __name__ == '__main__':
    main()

