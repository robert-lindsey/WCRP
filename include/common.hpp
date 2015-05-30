#ifndef COMMON_H
#define COMMON_H

#include <vector>
#include <cstdlib>
#include <cmath>
#include <set>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <iostream>
#include <cassert>
#include <map>
#include <ctime>

#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/unordered_map.hpp>

#define UNASSIGNED 0		// special flag to indicate a item currently has no table assignment
#define TOL .0000000000001	// used to check for equality of doubles
#define LOG_TOL -29.9336062089
#define ONEMINUSTOL (1.0 - TOL)

// hyperparameters for the gamma-distributed alpha'
#define HYPER_AP1 1.0	// shape
#define HYPER_AP2 100.0	// scale

struct bkt_parameters {
	double mu;	// probability of transitioning from unlearned to learned state 
	double psi;	// probability of starting in the learned state
	double pi1;	// probability of a correct response in the learned state
	double pi0;	// probability of a correct response in the unlearned state
};

#endif
