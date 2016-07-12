#!/usr/bin/env python
import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages

# function for the log of the prior probability
def logPrior(mu,tau_sqr):
	log_pri = -math.log(1000.0) - math.log(tau_sqr) - (0.5*math.log(2*math.pi)) - (0.9*math.log(1.0/tau_sqr)) - (0.5* (mu**2.0)/((1000.0*tau_sqr)**2.0)) - (.1/tau_sqr)
	return log_pri

# function for the lof of the likelihood 
def logLikelihood(psi_hat,sigma,mu,tau_sqr):
	n = len(psi_hat)
	log_l = np.zeros((n,1))
	for i in range(n):
		log_l[i] = (   -((mu**2.0) * (sigma[i]**2.0)) - ((psi_hat[i]**2.0) * tau_sqr) + ((( (mu*(sigma[i]**2.0)) + (psi_hat[i]*tau_sqr) )**2.0) / ((sigma[i]**2.0) + tau_sqr)) ) / (2.0*(sigma[i]**2.0)*tau_sqr)
		log_l[i] = log_l[i] - 0.5*( math.log(2.0*math.pi) + math.log((sigma[i]**2.0)+tau_sqr) ) 

	log_likeli = np.sum(log_l)
	return log_likeli

# function for the log of the posterior
def logPosterior(mu,tau_sqr,psi_hat,sigma):
	prior = logPrior(mu,tau_sqr)
	like = logLikelihood(psi_hat,sigma,mu,tau_sqr)
	log_post = prior + like
	
	return log_post

# symmetric, normal proposal function
def proposal(mu,tau_sqr,del_mu,del_tausqr):
	within = False
	while not within:	
		new_mu = np.random.normal(mu,del_mu)
		new_tausqr = np.random.normal(tau_sqr,del_tausqr)
		
		if (0 < new_tausqr):
			within = True 
	
	return new_mu,new_tausqr

# acceptance function
def acceptance(old_p,new_p):
	numer = math.exp(new_p)			# convert  log probabilities to regular probabilities
	denom = math.exp(old_p)
	if np.random.rand() < (numer/denom):
		accept = True
	else:
		accept = False

	return accept
	
# Metropolis MC driver function
def metropMCDriver(n_steps,initial,psi_hat,sigma,del_mu,del_tausqr):
	thin = 10
	num_succ = 0
	smpls = np.zeros((n_steps/thin,2))
	mu = initial[0]
	tau_sqr = initial[1]
	old_p = logPosterior(mu,tau_sqr,psi_hat,sigma)
	for i in range(n_steps):	
		new_mu,new_tausqr = proposal(mu,tau_sqr,del_mu,del_tausqr)
		new_p = logPosterior(new_mu,new_tausqr,psi_hat,sigma)
		acc = acceptance(old_p,new_p)
		if acc:
			old_p = new_p
			mu = new_mu
			tau_sqr = new_tausqr
			num_succ += 1

		if (i%thin) == 0:
			smpls[i/thin] = [mu,tau_sqr]


	acc_ratio = float(num_succ)/float(n_steps)
	
	return smpls,acc_ratio

# input data
psi_hat = [1.06,-0.1,0.62,0.02,1.07,-0.02,-0.12,-0.38,0.51,0.00,0.38,0.40]
sigma = [0.37,0.11,0.22,0.11,0.12,0.12,0.22,0.23,0.18,0.32,0.20,0.25]

# initialize MC parameters
n_steps = 100000;
initial = [1.0,1.0]
del_mu = .25
del_tausqr = .25
# run the MC 
smpls,acc_ratio = metropMCDriver(n_steps,initial,psi_hat,sigma,del_mu,del_tausqr)
# save results to file for viewing
outfile = open('lab3results.txt','w')
np.savetxt(outfile,smpls,'%20.10f \t %20.10f')
outfile.close()

log_oddsratio = smpls[100:len(smpls),0]		#discard the first 100 samples as burn-in
# plot histogram of the samples of mu
plt.subplot(211)
plt.hist(log_oddsratio,normed = True) 
plt.xlabel('mu: log odds ratio')
plt.title('{0} samples, delta mu = {1},delta tausqr = {2}, \n initial = [{3},{4}] , accept ratio = {5}'.format(len(log_oddsratio),del_mu,del_tausqr,initial[0],initial[1],acc_ratio))

#plot traceplot of MC chain
plt.subplot(212)
plt.plot(smpls[100:len(smpls),0],smpls[100:len(smpls),1],'k-')
plt.xlabel('mu: log odds ratio of population')
plt.ylabel('tau^2')
plt.show()

