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

#ifndef MAIN_CPP
#define MAIN_CPP

#include "common.hpp"
#include "MixtureWCRP.hpp"

using namespace std;


// reads a tab delimited file with the columns: student id, item id, skill id, recall success
// all ids are assumed to start at 0 and be contiguous
void load_dataset(const char * filename, vector<size_t> & provided_skill_assignments, vector< vector<bool> > & recall_sequences, vector< vector<size_t> > & item_sequences, size_t & num_students, size_t & num_items, size_t & num_skills) {

	num_students=0, num_items=0, num_skills=0;
	size_t student, item, skill, recall;

	ifstream in(filename);
	if (!in.is_open()) { 
		cerr << "couldn't open " << string(filename) << endl;
		exit(EXIT_FAILURE);
	}
	
	// figure out how many students, items, and skills there are
	while (in >> student >> item >> skill >> recall) {
		num_students = max(student+1, num_students);
		num_items = max(item+1, num_items);
		num_skills = max(skill+1, num_skills);
	}
	in.close();
	cout << "dataset has " << num_students << " students, " << num_items << " items, and " << num_skills << " expert-provided skills" << endl;

	// initialize
	provided_skill_assignments.resize(num_items, -1); // skill_assignments[item index] = skill index
	recall_sequences.resize(num_students);
	item_sequences.resize(num_students);

	// read the dataset
	in.open(filename);
	while (in >> student >> item >> skill >> recall) {
		recall_sequences[student].push_back(recall);
		item_sequences[student].push_back(item);
		provided_skill_assignments[item] = skill;
	}
	in.close();
}


void load_splits(const char * filename, vector<vector<size_t> > & fold_nums, size_t & num_folds, const size_t num_students) {

	ifstream in(filename);
	if (!in.is_open()) { 
		cerr << "couldn't open " << string(filename) << endl;
		exit(EXIT_FAILURE);
	}
	
	num_folds = 0;
	while(!in.eof()) {
		// read a line
		string line;
		getline(in, line);
		boost::trim(line);
		if (line.empty()) break;
	
		// split on whitespace
		vector<string> fields;
		boost::split(fields, line, boost::is_any_of(" \t"));
		assert(fields.size() == num_students);
		
		vector<size_t> replication_fold_nums(fields.size());
		for (size_t student = 0; student < fields.size(); student++) {
			replication_fold_nums[student] = boost::lexical_cast<size_t>(fields[student]);
			num_folds = max(replication_fold_nums[student]+1, num_folds);
		}
		fold_nums.push_back(replication_fold_nums);
	}
	
	cout << "# replications to run = " << fold_nums.size() << endl;
	cout << "# folds per replication = " << num_folds << endl;
}


int main(int argc, char ** argv) {

	namespace po = boost::program_options;

	string datafile, predfile, foldfile;
	int tmp_num_iterations, tmp_burn, tmp_num_subsamples;
	double init_beta, init_alpha_prime;
	bool infer_beta, infer_alpha_prime;

	// parse the command line arguments
	po::options_description desc("Allowed options");
	desc.add_options()
        ("help", "print help message")
		("datafile", po::value<string>(&datafile), "train the model on the given data file")
		("predfile", po::value<string>(&predfile), "file to put the posterior expected probability of recall for the heldout data")
		("foldfile", po::value<string>(&foldfile), "file with the training / test splits")
		("init_beta", po::value<double>(&init_beta), "initial value of beta")
		("fixed_alpha_prime", po::value<double>(&init_alpha_prime), "fixed value of alpha'")
		("infer_beta", "infer the value of beta")
		("num_iterations", po::value<int>(&tmp_num_iterations)->default_value(200), "number of iterations to run")
		("burn", po::value<int>(&tmp_burn)->default_value(100), "number of iterations to discard")
		("num_subsamples", po::value<int>(&tmp_num_subsamples)->default_value(2000), "number of samples to use when approximating marginal likelihood of new tables")
	;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (argc == 1 || vm.count("help")) {
		cout << desc << endl;
		return EXIT_SUCCESS;
	}

	if (vm.count("fixed_alpha_prime")) {
		assert(init_alpha_prime >= 0);
		infer_alpha_prime = false;
		cout << "the code will keep alpha' fixed at " << init_alpha_prime << endl;
	}
	else {
		init_alpha_prime = -1;
		infer_alpha_prime = true;
		cout << "the code will automatically infer the value of alpha'" << endl;
	}
	
	size_t num_iterations = (size_t) tmp_num_iterations;
	size_t burn = (size_t) tmp_burn;
	size_t num_subsamples = (size_t) tmp_num_subsamples;

	Random * generator = new Random(time(NULL));

	infer_beta = vm.count("infer_beta");
	if (infer_beta) cout << "the code will automatically infer the value of beta" << endl;
	else cout << "the code will keep beta fixed at " << init_beta << endl;
	
	assert(init_beta >= 0 && init_beta <= 1);
	assert(num_iterations >= 0);
	assert(num_iterations > burn);

	// load the dataset
	vector<size_t> provided_skill_assignments;
	vector< vector<bool> > recall_sequences; // recall_sequences[student][trial # i]  = recall success or failure of the ith trial we have for the student
	vector< vector<size_t> > item_sequences; // item_sequences[student][trial # i] = item corresponding to the ith trial we have for the student
	size_t num_students, num_items, num_skills_dataset;
	load_dataset(datafile.c_str(), provided_skill_assignments, recall_sequences, item_sequences, num_students, num_items, num_skills_dataset);
	assert(num_students > 0 && num_items > 0);

	// load the training-test splits
	vector<vector<size_t> > fold_nums;
	size_t num_folds;
	load_splits(foldfile.c_str(), fold_nums, num_folds, num_students);
	
	// create the file where we'll put our recall probability predictions 
	ofstream out_predictions(predfile.c_str(), ofstream::out);
	out_predictions << "replication\tfold\twas_heldout\tstudent_recalled\tprob_recall" << endl; // write the file header
	
	for (size_t replication = 0; replication < fold_nums.size(); replication++) {
		for (size_t test_fold = 0; test_fold < num_folds; test_fold++) {
		
			// figure out which students are in the training set for this replication-fold
			set<size_t> train_students; 
			for (size_t s = 0; s < num_students; s++) {
				if (fold_nums.at(replication).at(s) != test_fold || num_folds<=1) train_students.insert(s);
			}
			assert(!train_students.empty());
			
			// create the model
			MixtureWCRP model(generator, train_students, recall_sequences, item_sequences, provided_skill_assignments, init_beta, init_alpha_prime, num_students, num_items, num_subsamples);

			// run the sampler
			model.run_mcmc(num_iterations, burn, infer_beta, infer_alpha_prime);
			
			// save the posterior expected recall probability for each student-trial to file
			for (size_t student = 0; student < num_students; student++) {
				const bool was_heldout = !train_students.count(student);
				for (size_t trial = 0; trial < recall_sequences.at(student).size(); trial++) {
					const double mean_prob = model.get_estimated_recall_prob(student, trial);
					out_predictions << replication << "\t" << test_fold << "\t" << was_heldout << "\t" << recall_sequences.at(student).at(trial) << "\t" << mean_prob << endl;
				}
			}
		}
	}

	delete generator;
	return EXIT_SUCCESS;
}

#endif
