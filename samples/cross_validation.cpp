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

#ifndef CROSS_VALIDATION_CPP
#define CROSS_VALIDATION_CPP

#include "common.hpp"
#include "MixtureWCRP.hpp"

using namespace std;


int main(int argc, char ** argv) {

    namespace po = boost::program_options;

    string datafile, savefile, foldfile, expertfile;
    int tmp_num_iterations, tmp_burn, tmp_num_subsamples;
    double init_beta, init_alpha_prime;
    bool infer_beta, infer_alpha_prime;
    bool has_abilities, has_forgetting, use_auto_expertlabels, run_one_test_fold;
    
    // parse the command line arguments
    po::options_description desc("Allowed options");
    desc.add_options()
            ("help", "print help message")
            
            ("run_one_test_fold", po::value<bool>(&run_one_test_fold)->default_value(false), "(optional) run test fold 0 only in each replication")
            
            ("use_auto_expertlabels", po::value<bool>(&use_auto_expertlabels)->default_value(false), "(optional) generate expert labels where skill_id = problem_id")
            ("has_forgetting", po::value<bool>(&has_forgetting)->default_value(false), "(optional) support forgetting")
            ("has_abilities", po::value<bool>(&has_abilities)->default_value(false), "(optional) support student abilities")
            
            ("datafile", po::value<string>(&datafile), "(required) file containing the student recall data")
            
            ("savefile", po::value<string>(&savefile), "(required) file to put the posterior expected probability of recall for the heldout data")
            ("foldfile", po::value<string>(&foldfile), "(required) file with the training-test splits")
            ("expertfile", po::value<string>(&expertfile), "(optional) file containing the expert-provided skill labels")
            ("iterations", po::value<int>(&tmp_num_iterations)->default_value(1000), "(optional) number of iterations to run. i highly recommend you tune this parameter")
            ("burn", po::value<int>(&tmp_burn)->default_value(500), "(optional) number of iterations to discard. i highly recommend you tune this parameter")
            ("fix_alpha_prime", po::value<double>(&init_alpha_prime), "(optional) fix alpha' at the provided value instead of letting the model try to estimate it")
            ("fix_beta", po::value<double>(&init_beta), "(optional) fix beta at the provided value instead of giving it the Bayesian treatment")
            ("num_subsamples", po::value<int>(&tmp_num_subsamples)->default_value(2000), "number of samples to use when approximating marginal likelihood of new tables")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (argc == 1 || vm.count("help")) {
        cout << desc << endl;
        return EXIT_SUCCESS;
    }

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

    assert(init_beta >= 0 && init_beta <= 1);
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
    else if (use_auto_expertlabels) {
        for (int k = 0; k < num_items; ++k) 
            provided_skill_labels[k] = k;
    }
    else {
        // tell the model to ignore provided_skill_labels:
        init_beta = 0.0;
        infer_beta = false;
    }
    
        
    // load the training-test splits
    vector<vector<size_t> > fold_nums;
    size_t num_folds;
    load_splits(foldfile.c_str(), fold_nums, num_folds, num_students);

    // create the file where we'll put our recall probability predictions
    ofstream out_predictions(savefile.c_str(), ofstream::out);
    out_predictions << "replication\tfold\twas_heldout\tstudent_recalled\tprob_recall" << endl; // write the file header

    for (size_t replication = 0; replication < fold_nums.size(); replication++) {
        for (size_t test_fold = 0; test_fold < num_folds; test_fold++) {
            if (run_one_test_fold && test_fold > 0)
            	break;
            
            cout << "running replication " << replication << ", test fold " << test_fold << endl;
                
            // figure out which students are in the training set for this replication-fold
            set<size_t> train_students;
            for (size_t s = 0; s < num_students; s++) {
                if (fold_nums.at(replication).at(s) != test_fold || num_folds<=1) train_students.insert(s);
            }
            assert(!train_students.empty());

            // create the model
            MixtureWCRP model(generator, train_students, recall_sequences, item_sequences, provided_skill_labels, init_beta, init_alpha_prime, num_students, num_items, 
                num_subsamples, has_forgetting, has_abilities);

            // run the sampler
            model.run_mcmc(num_iterations, burn, infer_beta, infer_alpha_prime);

            // write the posterior expected recall probability for each student-trial to the output file
            
            for (size_t student = 0; student < num_students; student++) {
                const bool was_heldout = !train_students.count(student);
                for (size_t trial = 0; trial < recall_sequences.at(student).size(); trial++) {
                    const double mean_prob = model.get_estimated_recall_prob(student, trial);
                    //cout << replication << "\t" << test_fold << "\t" << was_heldout << "\t" << recall_sequences.at(student).at(trial) << "\t" << mean_prob << endl;
                
                    out_predictions << replication << "\t" << test_fold << "\t" << was_heldout << "\t" << recall_sequences.at(student).at(trial) << "\t" << mean_prob << endl;
                }
            }

            cout << "done" << endl << endl;
        }
    }

    delete generator;
    return EXIT_SUCCESS;
}

#endif
