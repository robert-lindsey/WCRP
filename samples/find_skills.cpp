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

#ifndef FIND_SKILLS_CPP
#define FIND_SKILLS_CPP

#include "common.hpp"
#include "MixtureWCRP.hpp"

using namespace std;


int main(int argc, char ** argv) {

	namespace po = boost::program_options;

	string datafile, savefile, expertfile;
	int tmp_num_iterations, tmp_burn, tmp_num_subsamples;
	double init_beta, init_alpha_prime;
	bool infer_beta, infer_alpha_prime, map_estimate;

	// parse the command line arguments
	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "print help message")
		("datafile", po::value<string>(&datafile), "(required) file containing the student recall data")
		("savefile", po::value<string>(&savefile), "(required) file to put the skill labels")
		("expertfile", po::value<string>(&expertfile), "(optional) file containing the expert-provided skill labels")
		("map_estimate", "(optional) save the MAP skill labels instead of all sampled skill labels")
		("iterations", po::value<int>(&tmp_num_iterations)->default_value(1000), "(optional but highly recommended) number of iterations to run. if you're not sure how to set it, use a large value")
		("burn", po::value<int>(&tmp_burn)->default_value(500), "(optional but highly recommended) number of iterations to discard. if you're not sure how to set it, use a large value (less than iterations)")
		("fix_alpha_prime", po::value<double>(&init_alpha_prime), "(optional) fix alpha' at the provided value instead of letting the model try to estimate it")
		("fix_beta", po::value<double>(&init_beta), "(optional) fix beta at the provided value instead of giving it the Bayesian treatment")
		("num_subsamples", po::value<int>(&tmp_num_subsamples)->default_value(2000), "number of auxiliary samples to use when approximating the marginal likelihood of new skills")
	;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (argc == 1 || vm.count("help")) {
		cout << desc << endl;
		return EXIT_SUCCESS;
	}

	map_estimate = vm.count("map_estimate");

	if (vm.count("fix_alpha_prime")) {
		assert(init_alpha_prime >= 0);
		infer_alpha_prime = false;
	}
	else {
		init_alpha_prime = -1;
		infer_alpha_prime = true;
	}
	
	size_t num_iterations = (size_t) tmp_num_iterations;
	size_t burn = (size_t) tmp_burn;
	size_t num_subsamples = (size_t) tmp_num_subsamples;

	Random * generator = new Random(time(NULL));

	if (vm.count("fix_beta")) {
		assert(init_beta >= 0 && init_beta <= 1);
		infer_beta = false;
	}
	else {
		// note: this will be overwritten if no expert-labels are provided 
		init_beta = .5; // arbitrary starting value < 1
		infer_beta = true;
	}

	assert(num_iterations >= 0);
	assert(num_iterations > burn);

	// load the dataset
	vector< vector<bool> > recall_sequences; // recall_sequences[student][trial # i]  = recall success or failure of the ith trial we have for the student
	vector< vector<size_t> > item_sequences; // item_sequences[student][trial # i] = item corresponding to the ith trial we have for the student
	size_t num_students, num_items, num_skills_dataset;
	load_student_data(datafile.c_str(), recall_sequences, item_sequences, num_students, num_items, num_skills_dataset);
	assert(num_students > 0 && num_items > 0);

	// load the expert-provided skill labels if possible 
	vector<size_t> provided_skill_labels(num_items, 0); 
	if (!expertfile.empty()) load_expert_labels(expertfile.c_str(), provided_skill_labels, num_items);
	else {
		// tell the model to ignore provided_skill_labels: 
		init_beta = 0.0;
		infer_beta = false;
	}
	
	// we'll let the model use all the students as training data: 
	set<size_t> train_students; 
	for (size_t s = 0; s < num_students; s++) train_students.insert(s);
	
	// create the model
	MixtureWCRP model(generator, train_students, recall_sequences, item_sequences, provided_skill_labels, init_beta, init_alpha_prime, num_students, num_items, num_subsamples);

	// run the sampler
	model.run_mcmc(num_iterations, burn, infer_beta, infer_alpha_prime);
	
	ofstream out_skills(savefile.c_str(), ofstream::out);
	if (map_estimate) { // save the most likely skill label
		vector<size_t> map_estimate = model.get_most_likely_skill_labels();
		assert(map_estimate.size() == num_items);
		for (size_t item = 0; item < num_items; item++) {
			out_skills << map_estimate.at(item);
			if (item == num_items - 1) out_skills << endl;
			else out_skills << " ";
		}
	}
	else { // save all sampled skill labels 
		vector< vector<size_t> > skill_samples = model.get_sampled_skill_labels();
		assert(!skill_samples.empty());
		for (size_t sample = 0; sample < skill_samples.size(); sample++) {
			assert(skill_samples.at(sample).size() == num_items);
			for (size_t item = 0; item < num_items; item++) {
				out_skills << skill_samples.at(sample).at(item);
				if (item == num_items - 1) out_skills << endl;
				else out_skills << " ";
			}
		}
	}

	delete generator;
	return EXIT_SUCCESS;
}

#endif
