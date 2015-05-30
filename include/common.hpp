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
	double prop0; // probability of a correct response in the unlearned state is prop0*pi1
				  //  it enforces Pr(correct | learned state) >= Pr(correct | unlearned state) 
};

#endif
