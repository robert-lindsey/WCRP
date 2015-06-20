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

#ifndef RANDOM_H
#define RANDOM_H

#include <map>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <ctime>
#include <vector>
#include <cassert>
#include <iostream>

using namespace std;

class Random {

 // this object provides an easy to use interface to the GSL scientific library's random functions
 public:

    // object constructor
    Random(const unsigned int seed);

    // object destructor
    ~Random();

    // randomly permute the given vector
    template<class T>
    void shuffle(vector<T> & vec) {
        for (size_t i = 0; i < vec.size(); i++) swap(vec[i], vec[sampleUniformDiscrete(vec.size())]);
    }

    // sample from a Beta distribution
    double sampleBeta(double alpha, double beta);

    // draw iid samples from a beta
    void sampleBeta(vector<double> & output, unsigned int num_vals, double alpha, double beta);

    // sample from a Bernoulli distribution
    bool sampleBernoulli(double p);

    // sample from a student's t-distribution
    double sampleStudentT(double dof);

    // sample from a geometric distribution
    unsigned int sampleGeometric(double p);

    // sample from a d-dimensional Discrete distribution with uniform event probabilities
    // returns an integer uniformly sampled on [0, d)
    unsigned int sampleUniformDiscrete(unsigned int d);

    // draw from a discrete distribution with the given event probabilities (must sum to 1)
    unsigned int sampleDiscrete(vector<double> & probs);

    // safely draw from a discrete distribution with LOG probabilities PROPORTIONAL to the given values (doesnt have to sum to 1)
    unsigned int sampleUnnormalizedDiscrete(vector<double> & unnormalized_log_probs);

    // draw from a uniform distribution over [0, upperBound)
    double sampleUniform(double upperBound = 1);

    // draw from a uniform[0, 1]
    double sampleUniform01() { return sampleUniform(1); }

    // draw iid samples from a uniform[0, 1]
    void sampleUniform01(unsigned int num_vals, vector<double> & draws);

    // draw from a normal distribution with given mean and standard deviation
    double sampleNormal(double mean, double stddev);

    // draw iid samples from a normal distribution with given mean and standard deviation
    void sampleNormal(vector<double> & output, unsigned int num_vals, double mean, double stddev);

    // draw from a gamma distribution
    double sampleGamma(double a, double b);

    // draw from a Dirichlet distribution
    void sampleDirichlet(vector<double> & hyper, vector<double> & output);

    // draw from a symmetric Dirichlet distribution
    void sampleSymmetricDirichlet(double hyper, unsigned int dims, vector<double> & output);

 protected:

    gsl_rng * rndgenerator;
    map<unsigned int, gsl_ran_discrete_t *> gslprobs;

};

#endif
