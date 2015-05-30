/*
The MIT License (MIT)

Copyright (c) 2015 Robert Lindsey

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef RANDOM_CPP
#define RANDOM_CPP

#include <algorithm>
#include <numeric>
#include <cmath>
#include "Random.hpp"

using namespace std;

// object constructor: seeds the random number generator with the current time
Random::Random(const unsigned int seed){
	srand(seed);
	const gsl_rng_type * T;
	gsl_rng_env_setup();
	T = gsl_rng_default;
	rndgenerator = gsl_rng_alloc (T);
	gsl_rng_set(rndgenerator, seed*1234);
}


// object destructor
Random::~Random(){
	gsl_rng_free(rndgenerator);
	rndgenerator = NULL;
	for (map<unsigned int, gsl_ran_discrete_t *>::iterator itr = gslprobs.begin(); itr != gslprobs.end(); itr++) gsl_ran_discrete_free(itr->second);
}
	
	
// sample from a Beta distribution
double Random::sampleBeta(double alpha, double beta){
	assert(alpha > 0 && beta > 0);
	return gsl_ran_beta(rndgenerator, alpha, beta);
}


double Random::sampleStudentT(double dof) {
	assert(dof > 0);
	return gsl_ran_tdist (rndgenerator, dof);
}

void Random::sampleBeta(vector<double> & output, unsigned int num_vals, double alpha, double beta) {
	output.resize(num_vals);
	for (unsigned int i=0; i < num_vals; i++) output[i] = sampleBeta(alpha, beta);
}


//sample from a Bernoulli distribution
bool Random::sampleBernoulli(double p){
	assert(p <= 1 && p >= 0);
	return gsl_ran_bernoulli(rndgenerator, p) > 0;
}


// sample from a d-dimensional Discrete distribution with uniform event probabilities
//  returns an integer uniformly sampled on [0, d)
unsigned int Random::sampleUniformDiscrete(unsigned int d){
	assert(d > 0);

	if (gslprobs.find(d) == gslprobs.end()){
		double * probs = new double[d];
		for (unsigned int i=0; i < d; i++) probs[i] = 1.0 / d;
		gsl_ran_discrete_t * gslprob = gsl_ran_discrete_preproc(d, probs);
		gslprobs[d] = gslprob; //save for later
		delete[] probs;
	}

	return gsl_ran_discrete(rndgenerator, gslprobs[d]);
}

// draw from a discrete distribution with the given event probabilities (must sum to 1)
unsigned int Random::sampleDiscrete(vector<double> & probs) {
	assert(!probs.empty());
	const double threshold = this->sampleUniform01();
	double total = 0.0;
	for (unsigned int i=0; i < probs.size(); i++) {
		total += probs.at(i);
		if (total >= threshold) return i;
	}
	assert(false);
}


// safely draw from a discrete distribution with LOG probabilities PROPORTIONAL to the given values (doesnt have to sum to 1)
unsigned int Random::sampleUnnormalizedDiscrete(vector<double> & unnormalized_log_probs) {
	assert(!unnormalized_log_probs.empty());

	// trick to prevent underflow. equivalent to multiplying all unnormalized probabilities by a constant, so doesn't change the result
	const double biggest_log_val = *max_element(unnormalized_log_probs.begin(), unnormalized_log_probs.end());
	double normalization_constant = 0.0;
	for (unsigned int i=0; i < unnormalized_log_probs.size(); i++) {
		unnormalized_log_probs[i] -= biggest_log_val;
		normalization_constant += exp(unnormalized_log_probs.at(i));
	}
	assert(normalization_constant > 0);

	const double threshold = normalization_constant * this->sampleUniform01();
	//cout << "threshold = " << threshold << "\tnorm constant = " << normalization_constant << endl;
	double total = 0.0;
	for (unsigned int i=0; i < unnormalized_log_probs.size(); i++) {
		total += exp(unnormalized_log_probs.at(i));
		if (total >= threshold) return i;
	}
	assert(false);
}


// draw from a uniform distribution over [0, upperBound)
double Random::sampleUniform(double upperBound){
	return gsl_ran_flat(rndgenerator, 0, upperBound);
}


// draw iid samples from a uniform[0, 1]
void Random::sampleUniform01(unsigned int num_vals, vector<double> & output) {
	output.resize(num_vals);
	for (unsigned int i=0; i < num_vals; i++) output[i] = sampleUniform01();
}


// draw from a normal distribution with given mean and standard deviation
double Random::sampleNormal(double mean, double stddev){
	return mean + gsl_ran_gaussian(rndgenerator, stddev);
}


// draw iid samples from a normal distribution with given mean and standard deviation
void Random::sampleNormal(vector<double> & output, unsigned int num_vals, double mean, double stddev) {
	output.resize(num_vals);
	for (unsigned int i=0; i < num_vals; i++) output[i] = sampleNormal(mean, stddev);
}


// draw from a gamma distribution
double Random::sampleGamma(double a, double b){
	return gsl_ran_gamma(rndgenerator, a, b);
}

// sample from a geometric distribution  (begins at 1)
unsigned int Random::sampleGeometric(double p) {
	return gsl_ran_geometric(rndgenerator, p);
}


// sample from a dirichlet distribution
void Random::sampleDirichlet(vector<double> & hyper, vector<double> & output) {
	double * alpha = new double[hyper.size()];
	for (unsigned int i=0; i < hyper.size(); i++) alpha[i] = hyper.at(i);
	double * tmp = new double[hyper.size()];
	gsl_ran_dirichlet(rndgenerator, hyper.size(), alpha, tmp);
	output.resize(hyper.size());
	for (unsigned int i=0; i < hyper.size(); i++) output[i] = tmp[i];
	delete alpha;
	delete tmp;
}


void Random::sampleSymmetricDirichlet(double hyper, unsigned int dims, vector<double> & output) {
	vector<double> alpha(dims, hyper);
	sampleDirichlet(alpha, output);
}

#endif
