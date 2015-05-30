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

#ifndef MIXTURE_WCRP_CPP
#define MIXTURE_WCRP_CPP

#include "MixtureWCRP.hpp"

using namespace std;

double vector_sum(const vector<double> & vec) {
	if (vec.empty()) return 0.0;
	else return accumulate(vec.begin(), vec.end(), 0.0);
}

double vector_mean(const vector<double> & vec) {
	assert(!vec.empty());
	return vector_sum(vec) / vec.size();
}

inline bool equals_zero(const double x) {
	return abs(x) <= TOL;
}

// log(proportional to a uniform prior on log(x) )
double log_loggamma_prior_density(const double x) {
	assert(x <= 0);
	return 0.0;
}

// log(proportional to a gamma prior on alpha')
double log_logalphaprime_prior_density(const double log_alpha_prime) {
	const double alpha_prime = exp(log_alpha_prime);
	return (HYPER_AP1 - 1.0) * log_alpha_prime - alpha_prime / HYPER_AP2;
}


/////////////////////////////////////////////////
//////////// WCRP equations /////////////////////
/////////////////////////////////////////////////


// log(proportional to equation 1 in the NIPS paper)
double log_old_table_probability(const size_t num_seated, const double K, const double log_gamma, const size_t num_expert_provided_skills) {
	const double gamma = exp(log_gamma);
	return -log(1.0 * num_expert_provided_skills) + log(1.0*num_seated) + log(K + (1.0 - K) * gamma) - log(1.0/num_expert_provided_skills + (1.0 - 1.0/num_expert_provided_skills) * gamma);
}


// log(proportional to equation 2 in the NIPS paper)
double log_new_table_probability(const double log_alpha_prime, const double log_gamma, const size_t num_expert_provided_skills) {
	return -log(1.0 * num_expert_provided_skills) + log_alpha_prime + log_gamma;
}


// compute the variable K as defined in equation 1 in the NIPS paper
// when generative_mode = false, this assumes that the item has not been assigned to a table yet
double MixtureWCRP::compute_K(const size_t item, const size_t table_id, const bool generative_mode) const {

	assert(generative_mode || seating_arrangement.at(item) == UNASSIGNED);
	const double gamma = exp(log_gamma); 	
	const size_t end_idx = (generative_mode) ? item : num_items;
	const size_t item_expert_label = provided_skill_assignments.at(item);

	// for each expert skill id that occurs at this table, count the number of items at this table with that id, n_k^j, for all k
	boost::unordered_map<size_t, int> counts; // mapping b/w expert skill id => # of items at this table with that id

	int max_count = 0; 
	for (size_t other_item = 0; other_item < end_idx; other_item++) {
		if (item != other_item && seating_arrangement.at(other_item) == table_id) { // if customer k is sitting at this table
			const size_t expert_label = provided_skill_assignments.at(other_item);
			if (counts.find(expert_label) == counts.end()) counts[expert_label] = 1;
			else counts[expert_label]++;
			
			if (counts[expert_label] > max_count) max_count = counts[expert_label];
		}
	}

	const bool has_item_expert_label = (counts.find(item_expert_label) != counts.end());
	const double numerator_K = (has_item_expert_label) ? pow(gamma, max_count - counts.at(item_expert_label)) : pow(gamma, max_count);
	double denominator_K =  (num_expert_provided_skills - counts.size()) * pow(gamma, max_count); // the n_k^j == 0 cases
	for (boost::unordered_map<size_t, int>::const_iterator count_itr = counts.begin(); count_itr != counts.end(); count_itr++) denominator_K += pow(gamma, max_count - count_itr->second);  // the n_k^j != 0 cases
	return numerator_K / denominator_K;
}

/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////


// object constructor
MixtureWCRP::MixtureWCRP(Random * generator, 
                         const set<size_t> & train_students, 
                         const set<size_t> & test_students, 
                         const vector< vector<bool> > & recall_sequences, 
                         const vector< vector<size_t> > & item_sequences, 
                         const vector<size_t> & provided_skill_assignments, 
                         const double beta, 
                         const double init_alpha_prime, 
                         const size_t num_students, 
                         const size_t num_items, 
                         const size_t num_subsamples) :
                         
 generator(generator), 
 train_students(train_students), 
 test_students(test_students), 
 recall_sequences(recall_sequences), 
 item_sequences(item_sequences), 
 provided_skill_assignments(provided_skill_assignments), 
 num_students(num_students), 
 num_items(num_items), 
 num_subsamples(num_subsamples), 
 log_gamma(log(1.0 - beta)), 
 num_used_skills(0), 
 tables_ever_instantiated(UNASSIGNED+1) {

	// for legacy reasons, we define gamma = 1.0 - beta and do inference on log_gamma 
	const double gamma = 1.0 - beta; 

	assert(!train_students.empty() );
	assert(!provided_skill_assignments.empty());
	assert(gamma >= 0 && gamma <= 1);
	
	use_expert_labels = equals_zero(gamma);
	num_expert_provided_skills = 1 + *max_element(provided_skill_assignments.begin(), provided_skill_assignments.end());

	// the variable all_items will be useful during gibbs sampling
	all_items.resize(num_items);
	for (size_t i = 0; i < num_items; i++) all_items[i] = i;
	
	// to avoid unnecessary work during MCMC, for each student-item pair, figure out the trial index it was first studied
	first_encounter.resize(num_students);
	item_and_recall_sequences.resize(num_students);
	trials_studied.resize(num_students);
	for (size_t student = 0; student < num_students; student++) {
		first_encounter[student].resize(num_items, item_sequences.at(student).size());
		item_and_recall_sequences[student].resize(item_sequences.at(student).size());
		trials_studied[student].resize(num_items);
		for (size_t trial = 0; trial < item_sequences.at(student).size(); trial++) {
			const size_t item = item_sequences.at(student).at(trial);
			assert(item < num_items);
			first_encounter[student][item] = min(trial, first_encounter.at(student).at(item));
			item_and_recall_sequences[student][trial] = make_pair(item, recall_sequences.at(student).at(trial));
			trials_studied[student][item].push_back(trial);
		}
	}

	// to avoid unnecessary work during MCMC, figure out which students studied which items in the training data
	// also compute some statistics of the dataset so we can make a simple baseline model
	vector<double> provided_skill_totals(num_expert_provided_skills, 0), provided_skill_counts(num_expert_provided_skills, 0);
	ever_studied.resize(num_students); // training data only
	pRT_samples.resize(num_students);
	for (size_t student = 0; student < num_students; student++) {
		pRT_samples[student].resize(recall_sequences.at(student).size());
		ever_studied[student].resize(num_items, false);
		if (train_students.count(student)) {
			for (size_t trial = 0; trial < item_sequences.at(student).size(); trial++) {
				const size_t item = item_sequences.at(student).at(trial);
				ever_studied[student][item] = true;
				provided_skill_totals[provided_skill_assignments.at(item)] += recall_sequences.at(student).at(trial);
				provided_skill_counts[provided_skill_assignments.at(item)]++;
			}
		}
	}

	// store the baseline model's constant predictions, among other things
	students_who_studied.resize(num_items);
	all_first_encounters.resize(num_items);
	for (size_t item = 0; item < num_items; item++) {
		for (size_t student = 0; student < num_students; student++) {
			if (ever_studied.at(student).at(item)) {
				students_who_studied[item].push_back(student);
				all_first_encounters[item].push_back(first_encounter.at(student).at(item));
			}
		}

		const size_t provided_skill_id = provided_skill_assignments.at(item);
		if (provided_skill_means.find(provided_skill_id) == provided_skill_means.end()) {
			if (provided_skill_counts[provided_skill_id] > 0) provided_skill_means[provided_skill_id] = provided_skill_totals.at(provided_skill_id) / provided_skill_counts.at(provided_skill_id);
			else provided_skill_means[provided_skill_id] = .5;
		}
	}

	// initialize alpha'
	if (init_alpha_prime < 0) log_alpha_prime = log(generator->sampleGamma(HYPER_AP1, HYPER_AP2)); // sample value
	else log_alpha_prime = log(init_alpha_prime); // fix value

	// initialize the seating arrangement to the expert provided skills
	seating_arrangement.resize(num_items, UNASSIGNED);
	set<size_t> skills_encountered;
	for (size_t item = 0; item < num_items; item++) {
		const size_t table_id = 1 + provided_skill_assignments.at(item);
		assign_item_to_table(item, table_id, !skills_encountered.count(table_id));
		skills_encountered.insert(table_id);
	}
	tables_ever_instantiated += num_expert_provided_skills + 1; // +1 necessary?
	
	// sanity check
	size_t num_missing = 0;
	for (size_t item = 0; item < num_items; item++) {
		if (all_first_encounters.at(item).empty()) num_missing++;
	}
	if (num_missing > 0) cerr << "warning: " << num_missing << " of " << num_items << " items have no training data" << endl;

	// precompute the marginal likelihood of each item if it were a singleton skill
	if (!use_expert_labels) {
		cout << "precomputing all possible singleton skill marginal likelihoods" << endl;
		singleton_skill_data_lp.resize(num_items);

		prior_samples.resize(num_subsamples);
		for (size_t subsample = 0; subsample < num_subsamples; subsample++) draw_bkt_param_prior(prior_samples[subsample]);

		for (size_t item = 0; item < num_items; item++) {
			const vector<size_t> & affected_students = students_who_studied.at(item);
			const vector<size_t> & first_exposures = all_first_encounters.at(item);
			const size_t cur_table_id = seating_arrangement.at(item);

			// temporarily unassign the item from its table
			const bool deleted_table = remove_item_from_table(item, cur_table_id);

			// create a singleton skill with this item
			const size_t tmp_table_id = tables_ever_instantiated++;
			assign_item_to_table(item, tmp_table_id, true);

			// record the log likelihood for this singleton skill under each draw from the prior
			singleton_skill_data_lp[item].resize(num_subsamples);
			for (size_t subsample = 0; subsample < num_subsamples; subsample++) {
				parameters[tmp_table_id] = prior_samples[subsample];
				singleton_skill_data_lp[item][subsample] = skill_log_likelihood(tmp_table_id, affected_students, first_exposures);
			}

			// delete the singleton skill
			remove_item_from_table(item, tmp_table_id);

			// reassign the item to its original table
			if (!deleted_table) assign_item_to_table(item, cur_table_id, false);
			else assign_item_to_table(item, tables_ever_instantiated++, true);
		}
	}
}


// object destructor
MixtureWCRP::~MixtureWCRP() {
	// close any open file handles 
	if (outfile_meta.is_open()) outfile_meta.close();
	if (outfile_predictions.is_open()) outfile_predictions.close();
	if (outfile_skills.is_open()) outfile_skills.close();
}


// returns true if student studied any of the provided items
bool MixtureWCRP::studied_any_of(const size_t student, const vector<size_t> & items) const {
	for (vector<size_t>::const_iterator item_itr = items.begin(); item_itr != items.end(); item_itr++) {
		if (ever_studied.at(student).at(*item_itr)) return true;
	}
	return false;
}


// draw each of the BKT parameters uniformly at random on [TOL, 1 - TOL]
// (BKT breaks down if the parameters are ever actually 0 or 1)
void MixtureWCRP::draw_bkt_param_prior(struct bkt_parameters & params) const {
	params.psi = TOL + (ONEMINUSTOL - TOL) * generator->sampleUniform01();
	params.mu = TOL + (ONEMINUSTOL - TOL) * generator->sampleUniform01();
	params.pi1 = TOL + (ONEMINUSTOL - TOL) * generator->sampleUniform01();
	params.pi0 = TOL + (ONEMINUSTOL - TOL) * generator->sampleUniform01();
}


void MixtureWCRP::assign_item_to_table(const size_t item, const size_t table_id, const bool is_new_table) {
	if (is_new_table) {
		struct bkt_parameters new_params;
		draw_bkt_param_prior(new_params);
		parameters[table_id] = new_params;

		seating_arrangement[item] = table_id;
		table_sizes[table_id] = 1;
		extant_tables.insert(table_id);
		num_used_skills++;

		// record the trial #'s for each student who studied this singleton skill
		trial_lookup[table_id] = boost::unordered_map<size_t, vector<size_t> >();
		for (vector<size_t>::const_iterator student_itr = students_who_studied.at(item).begin(); student_itr != students_who_studied.at(item).end(); student_itr++) {
			trial_lookup[table_id][*student_itr] = trials_studied[*student_itr][item];
		}
	}
	else { // sit at existing table
		seating_arrangement[item] = table_id;
		table_sizes[table_id]++;

		// update the trial #'s for each student for this skill
		for (vector<size_t>::const_iterator student_itr = students_who_studied.at(item).begin(); student_itr != students_who_studied.at(item).end(); student_itr++) {
			if (trial_lookup[table_id].find(*student_itr) == trial_lookup[table_id].end()) trial_lookup[table_id][*student_itr] = trials_studied[*student_itr][item]; // this student hadn't previously had any items assigned to this skill, but now does
			else {
				// this student had previously had at least one item assigned to this skill
				// merge trials_studied[*student_itr][item] into trial_lookup[table_id][*student_itr]
				vector<size_t> tmp;
				tmp.reserve(trial_lookup[table_id][*student_itr].size() + trials_studied[*student_itr][item].size());
				merge(trial_lookup[table_id][*student_itr].begin(), trial_lookup[table_id][*student_itr].end(), trials_studied[*student_itr][item].begin(), trials_studied[*student_itr][item].end(), std::back_inserter(tmp));
				trial_lookup[table_id][*student_itr].swap(tmp);
			}
		}
	}

	assert(num_used_skills == table_sizes.size());
}


// returns true if we deleted the table too
bool MixtureWCRP::remove_item_from_table(const size_t item, const size_t table_id) {
	table_sizes[table_id]--;
	seating_arrangement[item] = UNASSIGNED;

	if (table_sizes[table_id] == 0) {
		table_sizes.erase(table_id);
		parameters.erase(table_id);
		extant_tables.erase(table_id);
		num_used_skills--;

		trial_lookup.erase(table_id);
		return true;
	}

	// update the trial #'s for each student for this skill
	for (vector<size_t>::const_iterator student_itr = students_who_studied.at(item).begin(); student_itr != students_who_studied.at(item).end(); student_itr++) {
		// remove trials_studied[*student_itr][item] from trial_lookup[table_id][*student_itr]

		const size_t final_size = trial_lookup[table_id][*student_itr].size() - trials_studied[*student_itr][item].size();
		if (final_size == 0) {
			// the student now has no items assigned to this skill
			trial_lookup[table_id].erase(*student_itr);
		}
		else {
			vector<size_t> tmp;
			tmp.reserve(final_size);
			size_t num_ignored = 0;
			for (vector<size_t>::const_iterator itr = trial_lookup[table_id][*student_itr].begin(); itr != trial_lookup[table_id][*student_itr].end(); itr++) {
				if ( (num_ignored < trials_studied[*student_itr][item].size() && *itr != trials_studied[*student_itr][item][num_ignored]) || num_ignored >= trials_studied[*student_itr][item].size()) tmp.push_back(*itr);
				else num_ignored++;
			}
			assert(tmp.size() == final_size);
			trial_lookup[table_id][*student_itr].swap(tmp);
		}
	}

	return false;
}


// calculate the data log likelihood for the skill (table_id) across all students
// this version of the function is only used in updating BKT parameters
double MixtureWCRP::skill_log_likelihood(const size_t table_id, const vector<size_t> & affected_students, const vector<size_t> & first_exposures) const {

	double skill_log_lik = 0.0;

	// define some constants for notational clarity
	const struct bkt_parameters & skill_params = parameters.at(table_id);
	const double skill_pi1 = skill_params.pi1;
	const double skill_pi0 = skill_pi1 *  skill_params.pi0;
	const double skill_mu = skill_params.mu;
	const double skill_psi = skill_params.psi;

	// for each student who ever practiced this skill
	for (size_t k = 0; k < affected_students.size(); k++) {

		const size_t student = affected_students.at(k);
		const size_t start_trial = first_exposures.at(k);
		double student_skill_log_lik = 0.0;

		const vector< pair<size_t, bool> > & recall_items = item_and_recall_sequences.at(student);
		double cur_p_hat = skill_psi;

		// for each trial of this skill
		for (vector<size_t>::const_iterator trial_idx_itr = trial_lookup.at(table_id).at(student).begin(); trial_idx_itr != trial_lookup.at(table_id).at(student).end(); trial_idx_itr++) {
			const pair<size_t, bool> & trial_pair = recall_items.at(*trial_idx_itr);
			if (trial_pair.second) { // the student responded correctly
				if (*trial_idx_itr >= start_trial) student_skill_log_lik += log(skill_pi0 * (1.0 - cur_p_hat) + skill_pi1 * cur_p_hat);
				cur_p_hat = (skill_pi1 * cur_p_hat + skill_mu * skill_pi0 * (1.0 - cur_p_hat)) / (skill_pi1 * cur_p_hat + skill_pi0 * (1.0 - cur_p_hat));
			}
			else { // the student responded incorrectly
				if (*trial_idx_itr >= start_trial) student_skill_log_lik += log(1.0 - (skill_pi0 * (1.0 - cur_p_hat) + skill_pi1 * cur_p_hat));
				cur_p_hat = ((1.0 - skill_pi1) * cur_p_hat + skill_mu * (1.0 - skill_pi0) * (1.0 - cur_p_hat)) / ((1.0 - skill_pi1) * cur_p_hat + (1.0 - skill_pi0) * (1.0 - cur_p_hat));
			}
		}

		assert(isfinite(student_skill_log_lik));
		if (student_skill_log_lik > 0.0) student_skill_log_lik = 0.0; // occasional minor numerical issue because of my caching trick
		skill_log_lik += student_skill_log_lik;
	}

	assert(isfinite(skill_log_lik));
	assert(skill_log_lik <= TOL);
	skill_log_lik = min(0.0, skill_log_lik);
	return skill_log_lik;
}


// calculate the data log likelihood for the skill (table_id) across all students
//   for each student s, only the part of the log likelihood occurring on or after first_exposures[s] is included in the calculation
// for speed, this function uses each student's precomputed BKT sufficient statistic (init_p_hat) up to the first encounter of item
// used by MixtureWCRP::gibbs_resample_skill
double MixtureWCRP::skill_log_likelihood(const size_t table_id, const vector<size_t> & affected_students, const vector<size_t> & first_exposures, const vector< boost::unordered_map<size_t, double> > & init_p_hat) const {

	// if non-existent table, return 0
	if (table_sizes.find(table_id) == table_sizes.end() || table_sizes.at(table_id) == 0) return 0.0;

	double skill_log_lik = 0.0;

	// for convenience
	const struct bkt_parameters & skill_params = parameters.at(table_id);
	const double skill_pi1 = skill_params.pi1;
	const double skill_pi0 = skill_pi1 *  skill_params.pi0;
	const double skill_mu = skill_params.mu;

	for (size_t student_idx = 0; student_idx < affected_students.size(); student_idx++) {

		const size_t student = affected_students.at(student_idx);

		if (trial_lookup.at(table_id).find(student) == trial_lookup.at(table_id).end()) {
			// we likely unassigned the only item this student had in this skill, so they have no trials relevant to this skill....
			continue;
		}

		const size_t start_trial = first_exposures.at(student_idx);
		double student_skill_log_lik = 0.0;

		// define some references for convenience:
		const vector< pair<size_t, bool> > & recall_items = item_and_recall_sequences.at(student);
		double cur_p_hat = init_p_hat.at(student_idx).at(table_id);

		// for each trial of this skill
		for (vector<size_t>::const_iterator trial_idx_itr = trial_lookup.at(table_id).at(student).begin(); trial_idx_itr != trial_lookup.at(table_id).at(student).end(); trial_idx_itr++) {
			if (*trial_idx_itr >= start_trial) {
				const pair<size_t, bool> & trial_pair = recall_items.at(*trial_idx_itr);
				if (trial_pair.second) { // the student responded correctly
					student_skill_log_lik += log(skill_pi0 * (1.0 - cur_p_hat) + skill_pi1 * cur_p_hat);
					cur_p_hat = (skill_pi1 * cur_p_hat + skill_mu * skill_pi0 * (1.0 - cur_p_hat)) / (skill_pi1 * cur_p_hat + skill_pi0 * (1.0 - cur_p_hat));
				}
				else { // the student responded incorrectly
					student_skill_log_lik += log(1.0 - (skill_pi0 * (1.0 - cur_p_hat) + skill_pi1 * cur_p_hat));
					cur_p_hat = ((1.0 - skill_pi1) * cur_p_hat + skill_mu * (1.0 - skill_pi0) * (1.0 - cur_p_hat)) / ((1.0 - skill_pi1) * cur_p_hat + (1.0 - skill_pi0) * (1.0 - cur_p_hat));
				}
			}
		}

		assert(isfinite(student_skill_log_lik));
		if (student_skill_log_lik > 0.0) student_skill_log_lik = 0.0; // occasional minor numerical issue because of my caching trick
		skill_log_lik += student_skill_log_lik;
	}

	assert(isfinite(skill_log_lik));
	assert(skill_log_lik <= 0.0);
	return skill_log_lik;
}


// resample the skill assignment (table) for this item (customer)
// see algorithm 8 from http://www.stat.purdue.edu/~rdutta/24.PDF
void MixtureWCRP::gibbs_resample_skill(const size_t item) {

	const size_t cur_table_id = seating_arrangement.at(item);
	const vector<size_t> & affected_students = students_who_studied.at(item);
	const vector<size_t> & first_exposures = all_first_encounters.at(item);

	// unassign the item's skill label
	remove_item_from_table(item, cur_table_id);

	// precompute each student's BKT sufficient statistic up to the first encounter of item
	vector< boost::unordered_map<size_t, double> > p_hat(affected_students.size());
	for (size_t student_idx = 0; student_idx < affected_students.size(); student_idx++) cache_p_hat(affected_students.at(student_idx), first_exposures.at(student_idx), p_hat[student_idx]);

	vector<double> data_lp_with_item, data_lp_without_item, seating_lp;

	// preallocate memory
	const size_t final_size = extant_tables.size() + num_subsamples;
	data_lp_with_item.reserve(final_size);
	data_lp_without_item.reserve(final_size);
	seating_lp.reserve(final_size);

	// compute the data log likelihood (of affected students) for each extant skill with and without item being assigned to it
	// also compute the log probability of sitting here
	vector<size_t> keys;
	keys.reserve(extant_tables.size());
	for (set<size_t>::const_iterator table_itr = extant_tables.begin(); table_itr != extant_tables.end(); table_itr++) { // note: extant_tables won't change in this loop
		// data likelihood; with
		assign_item_to_table(item, *table_itr, false);
		data_lp_with_item.push_back(skill_log_likelihood(*table_itr, affected_students, first_exposures, p_hat));
		
		// data likelihood; without
		remove_item_from_table(item, *table_itr);
		data_lp_without_item.push_back(skill_log_likelihood(*table_itr, affected_students, first_exposures, p_hat));
		
		// seating probability
		const double K = compute_K(item, *table_itr, false);
		seating_lp.push_back(log_old_table_probability(table_sizes.at(*table_itr), K, log_gamma, num_expert_provided_skills));

		keys.push_back(*table_itr);
	}

	// use the precomputed marginal likelihoods for calculating the new seating prob
	data_lp_with_item.insert(data_lp_with_item.end(), singleton_skill_data_lp.at(item).begin(), singleton_skill_data_lp.at(item).end());
	data_lp_without_item.resize(data_lp_without_item.size() + num_subsamples, 0.0);
	seating_lp.resize(seating_lp.size() + num_subsamples, log_new_table_probability(log_alpha_prime, log_gamma, num_expert_provided_skills) - log(1.0*num_subsamples));

	assert(data_lp_with_item.size() == num_used_skills + num_subsamples);
	assert(data_lp_without_item.size() == num_used_skills + num_subsamples);
	assert(seating_lp.size() == num_used_skills + num_subsamples);
	
	// consider assigning every possible skill label to this item
	vector<double> proportional_log_probs(seating_lp.size());
	for (size_t event = 0; event < seating_lp.size(); event++) proportional_log_probs[event] = seating_lp.at(event) + data_lp_with_item.at(event) - data_lp_without_item.at(event);

	// draw a new skill label
	const size_t num_extant_tables = extant_tables.size();
	const size_t drawn_event = (size_t) generator->sampleUnnormalizedDiscrete(proportional_log_probs);
	if (drawn_event >= num_extant_tables) { // decided to sit at new table
		assign_item_to_table(item, tables_ever_instantiated++, true); // sit down
		const size_t chosen_subsample = drawn_event - num_extant_tables;
		parameters[seating_arrangement.at(item)] = prior_samples.at(chosen_subsample); // assign parameters
	}
	else { // decided to sit at old table
		const size_t table_id = keys.at(drawn_event);
		assign_item_to_table(item, table_id, false);
	}
}


// return the log joint probability of this WCRP seating arrangement
// TODO: clean up
double MixtureWCRP::log_seating_prob() const {

	double log_prob = 0.0;
	boost::unordered_map<size_t, size_t> table_counts_so_far;

	for (size_t item = 0; item < num_items; item++) {
		const size_t chosen_table_id = seating_arrangement.at(item);
		double chosen_proportional_prob;
		vector<double> proportional_probs;
		proportional_probs.reserve(table_counts_so_far.size() + 1);
		bool chose_old = false;

		// figure out which tables exist if only items 0...item - 1 have sat down
		for (boost::unordered_map<size_t, size_t>::const_iterator table_itr = table_counts_so_far.begin(); table_itr != table_counts_so_far.end(); table_itr++) {
			const double K = compute_K(item, table_itr->first, true);
			const double prob = exp(log_old_table_probability(table_itr->second, K, log_gamma, num_expert_provided_skills));
			proportional_probs.push_back(prob);
			if (table_itr->first == chosen_table_id) {
				chosen_proportional_prob = prob;
				chose_old = true;
			}
		}

		// new table probability
		proportional_probs.push_back(exp(log_new_table_probability(log_alpha_prime, log_gamma, num_expert_provided_skills)));
		if (!chose_old) chosen_proportional_prob = proportional_probs.at(proportional_probs.size() - 1);

		log_prob += log(chosen_proportional_prob) - log(vector_sum(proportional_probs));

		if (table_counts_so_far.find(chosen_table_id) == table_counts_so_far.end()) table_counts_so_far[chosen_table_id] = 1;
		else table_counts_so_far[chosen_table_id]++;
	}

	return log_prob;
}


// returns log Pr(recall observations for all students | chain state)
double MixtureWCRP::full_data_log_likelihood(const bool is_training, size_t & trials_included) const {
	double ll = 0.0;
	trials_included = 0;
	for (size_t student = 0; student < num_students; student++) {
		if ( (is_training && train_students.count(student)) || (!is_training && test_students.count(student))) {
			size_t num_trials = 0;
			ll += data_log_likelihood(student, 0, num_trials);
			trials_included += num_trials;
		}
	}
	return ll;
}


double MixtureWCRP::data_log_likelihood(const vector<size_t> & students, const vector<size_t> & first_exposures) const {
	double ll = 0.0;
	size_t n = 0;
	for (size_t k = 0; k < students.size(); k++) ll += data_log_likelihood(students.at(k), first_exposures.at(k), n);
	return ll;
}


void MixtureWCRP::record_sample(const size_t replication, const size_t test_fold, const size_t iter, const size_t burn, const bool infer_gamma, const double elapsed_time, const double train_ll, const double test_ll, const size_t train_n, const size_t test_n, const size_t num_skills, const double cur_seating_lp) {

	// output some summary information
	outfile_meta << replication << " " << test_fold << " " << iter << " " << infer_gamma << " " << elapsed_time << " " << train_ll << " " << test_ll << " " << train_n << " " << test_n << " " << log_gamma << " " << log_alpha_prime << " " << num_skills << " " << cur_seating_lp << endl;

	if (iter >= burn) {
		// output the model predictions for the dataset
		for (size_t student = 0; student < num_students; student++) {
			// define some references for convenience:
			const vector<bool> & recall_sequence = recall_sequences.at(student);
			const vector<size_t> & item_sequence = item_sequences.at(student);

			// initialize p_hat
			boost::unordered_map<size_t, double> p_hat;
			for (boost::unordered_map<size_t, struct bkt_parameters>::const_iterator table_itr = parameters.begin(); table_itr != parameters.end(); table_itr++) p_hat[table_itr->first] = table_itr->second.psi;

			for (size_t trial = 0; trial < recall_sequence.size(); trial++) {

				// define some variables for notational clarity
				const bool did_recall = recall_sequence.at(trial);
				const size_t table_id = seating_arrangement.at(item_sequence.at(trial));
				const struct bkt_parameters & skill_params = parameters.at(table_id);
				const double skill_pi1 = skill_params.pi1;
				const double skill_pi0 = skill_pi1 *  skill_params.pi0;
				const double skill_mu = skill_params.mu;
				const double cur_p_hat = p_hat.at(table_id);

				pRT_samples[student][trial].push_back(skill_pi0 * (1.0 - cur_p_hat) + skill_pi1 * cur_p_hat); // record prediction

				if (did_recall) p_hat[table_id] = (skill_pi1 * cur_p_hat + skill_mu * skill_pi0 * (1.0 - cur_p_hat)) / (skill_pi1 * cur_p_hat + skill_pi0 * (1.0 - cur_p_hat));
				else p_hat[table_id] = ((1.0 - skill_pi1) * cur_p_hat + skill_mu * (1.0 - skill_pi0) * (1.0 - cur_p_hat)) / ((1.0 - skill_pi1) * cur_p_hat + (1.0 - skill_pi0) * (1.0 - cur_p_hat));
			}
		}
	}
}


// append the current skill assignments to the file handle outfile_skills
void MixtureWCRP::record_skill_assignments(const size_t replication, const size_t test_fold) {
	
	boost::unordered_map<size_t, int> skill_labels;
	outfile_skills << replication << " " << test_fold << " ";
	for (size_t item = 0; item < num_items; item++) {
		const size_t table_id = seating_arrangement.at(item);
		if (skill_labels.find(table_id) == skill_labels.end()) skill_labels[table_id] = skill_labels.size();
		
		outfile_skills << skill_labels.at(table_id);
		if (item != num_items - 1) outfile_skills << " ";
		else outfile_skills << endl;
	}
}


// sets p_hat to the across-skill state right before the trial = first_exposure
// used to reduce redundant computation during gibbs sampling
void MixtureWCRP::cache_p_hat(const size_t student, const size_t end_trial, boost::unordered_map<size_t, double> & p_hat) const {

	// define some references for convenience:
	const vector<bool> & recall_sequence = recall_sequences.at(student);
	const vector<size_t> & item_sequence = item_sequences.at(student);

	// initialize p_hat
	for (boost::unordered_map<size_t, struct bkt_parameters>::const_iterator table_itr = parameters.begin(); table_itr != parameters.end(); table_itr++) p_hat[table_itr->first] = table_itr->second.psi;

	for (size_t trial = 0; trial < end_trial; trial++) {

		// define some constants for notational clarity
		const bool did_recall = recall_sequence.at(trial);
		const size_t table_id = seating_arrangement.at(item_sequence.at(trial));
		const struct bkt_parameters & skill_params = parameters.at(table_id);
		const double skill_pi1 = skill_params.pi1;
		const double skill_pi0 = skill_pi1 *  skill_params.pi0;
		const double skill_mu = skill_params.mu;
		const double cur_p_hat = p_hat.at(table_id);
		const double one_minus_cur_p_hat = 1.0 - cur_p_hat;

		// update
		if (did_recall) p_hat[table_id] = (skill_pi1 * cur_p_hat + skill_mu * skill_pi0 * one_minus_cur_p_hat) / (skill_pi1 * cur_p_hat + skill_pi0 * one_minus_cur_p_hat);
		else p_hat[table_id] = ((1.0 - skill_pi1) * cur_p_hat + skill_mu * (1.0 - skill_pi0) * one_minus_cur_p_hat) / ((1.0 - skill_pi1) * cur_p_hat + (1.0 - skill_pi0) * one_minus_cur_p_hat);
	}
}


// returns log Pr(recall sequences for the student for trial >= start_trial | chain state)
double MixtureWCRP::data_log_likelihood(const size_t student, const size_t start_trial, size_t & num_trials) const {

	double log_lik = 0.0;

	// define some references for convenience:
	const vector<bool> & recall_sequence = recall_sequences.at(student);
	const vector<size_t> & item_sequence = item_sequences.at(student);
	num_trials = item_sequence.size(); // only used by full_data_log_likelihood. not important to the sampler

	boost::unordered_map<size_t, double> p_hat; // psi is a vector of length max_num_skills
	for (boost::unordered_map<size_t, struct bkt_parameters>::const_iterator table_itr = parameters.begin(); table_itr != parameters.end(); table_itr++) p_hat[table_itr->first] = table_itr->second.psi;

	for (size_t trial = 0; trial < num_trials; trial++) {

		// define some constants for notational clarity
		const bool did_recall = recall_sequence.at(trial);
		const size_t table_id = seating_arrangement.at(item_sequence.at(trial));
		const struct bkt_parameters & skill_params = parameters.at(table_id);
		const double skill_pi1 = skill_params.pi1;
		const double skill_pi0 = skill_pi1 *  skill_params.pi0;
		const double skill_mu = skill_params.mu;
		const double cur_p_hat = p_hat.at(table_id);

		// prediction
		const double pRT = skill_pi0 * (1.0 - cur_p_hat) + skill_pi1 * cur_p_hat;

		// update
		if (did_recall) {
			if (trial >= start_trial) log_lik += log(pRT);
			p_hat[table_id] = (skill_pi1 * cur_p_hat + skill_mu * skill_pi0 * (1.0 - cur_p_hat)) / (skill_pi1 * cur_p_hat + skill_pi0 * (1.0 - cur_p_hat));
		}
		else {
			if (trial >= start_trial) log_lik += log(1.0 - pRT);
			p_hat[table_id] = ((1.0 - skill_pi1) * cur_p_hat + skill_mu * (1.0 - skill_pi0) * (1.0 - cur_p_hat)) / ((1.0- skill_pi1) * cur_p_hat + (1.0 - skill_pi0) * (1.0 - cur_p_hat));
		}
	}

	assert(isfinite(log_lik));
	assert(log_lik <= 0.0);
	return log_lik;
}


// perform a slice sampling update on the provided BKT parameter, assuming a uniform prior
double MixtureWCRP::slice_resample_bkt_parameter(const size_t table_id, double * param, const vector<size_t> & students_to_include, const vector<size_t> & first_exposures, const double cur_ll) {
	
	const double lower_bound = TOL;
	const double upper_bound = ONEMINUSTOL;
	const double initial_bracket_width = (upper_bound - lower_bound) / 10.0;
	const double cur_val = *param;
	const double jittered_cur_ll = cur_ll + log(generator->sampleUniform01());
	const double split_location = generator->sampleUniform01();
	double x_l = max(lower_bound, cur_val - split_location * initial_bracket_width);
	double x_r = min(upper_bound, cur_val + (1.0 - split_location) * initial_bracket_width);

	*param = x_l;
	while (x_l >= lower_bound && skill_log_likelihood(table_id, students_to_include, first_exposures) > jittered_cur_ll) {
		x_l -= initial_bracket_width;
		*param = x_l;
	}
	x_l = max(x_l, lower_bound);

	*param = x_r;
	while (x_r <= upper_bound && skill_log_likelihood(table_id, students_to_include, first_exposures) > jittered_cur_ll) {
		x_r += initial_bracket_width;
		*param = x_r;
	}
	x_r = min(x_r, upper_bound);

	while (true) {
		*param = x_l + (x_r - x_l) * generator->sampleUniform01();
		const double proposal_ll = skill_log_likelihood(table_id, students_to_include, first_exposures);
		if (proposal_ll > jittered_cur_ll) return proposal_ll;
		else {
			if (*param > cur_val) x_r = *param;
			else if (*param < cur_val) x_l = *param;
			else return proposal_ll;
		}
	}
}


// perform a slice sampling update on the provided WCRP hyperparameter 
double MixtureWCRP::slice_resample_wcrp_param(double * param, const double cur_seating_lp, const double lower_bound, const double upper_bound, const double initial_bracket_width, prior_log_density_fn prior_lp) {

	const double cur_val = *param;
	const double jittered_cur_ll = cur_seating_lp + prior_lp(cur_val) + log(generator->sampleUniform01());
	const double split_location = generator->sampleUniform01();
	double x_l = max(lower_bound, cur_val - split_location * initial_bracket_width);
	double x_r = min(upper_bound, cur_val + (1.0 - split_location) * initial_bracket_width);

	*param = x_l;
	while (x_l >= lower_bound && log_seating_prob() + prior_lp(*param) > jittered_cur_ll) {
		x_l -= initial_bracket_width;
		*param = x_l;
	}
	x_l = max(x_l, lower_bound);

	*param = x_r;
	while (x_r <= upper_bound && log_seating_prob() + prior_lp(*param) > jittered_cur_ll) {
		x_r += initial_bracket_width;
		*param = x_r;
	}
	x_r = min(x_r, upper_bound);

	while (true) {
		*param = x_l + (x_r - x_l) * generator->sampleUniform01();
		const double proposal_ll = log_seating_prob();
		if (proposal_ll + prior_lp(*param) > jittered_cur_ll) return proposal_ll;
		else {
			if (*param > cur_val) x_r = *param;
			else if (*param < cur_val) x_l = *param;
			else return proposal_ll;
		}
	}
}


void MixtureWCRP::run_mcmc(const string outfilename, const size_t replication, const size_t test_fold, const size_t num_iterations, const size_t burn, const bool infer_gamma, const bool infer_alpha_prime, const bool dump_skills) {

	const string outfilename1 = outfilename + "_meta.txt";
	const string outfilename2 = outfilename + "_predictions.txt";
	const string outfilename3 = outfilename + "_skills.txt";
	if (replication == 0 && test_fold == 0) { // create new file
		outfile_meta.open(outfilename1.c_str(), ofstream::out);
		outfile_predictions.open(outfilename2.c_str(), ofstream::out);
		if (dump_skills) outfile_skills.open(outfilename3.c_str(), ofstream::out);
	}
	else { // append to file
		outfile_meta.open(outfilename1.c_str(), ofstream::app);
		outfile_predictions.open(outfilename2.c_str(), ofstream::app);
		if (dump_skills) outfile_skills.open(outfilename3.c_str(), ofstream::app);
	}

	for (size_t iter = 0; iter < num_iterations; iter++) {
		cout << "REPLICATION " << (replication+1) << ", TEST FOLD " << (test_fold+1) << ", ITERATION " << (iter+1) << " OF " << num_iterations << endl;

		clock_t begin = clock();
		
		// update alpha' and gamma
		cout << "  resampling WCRP hyperparameters" << endl;
		double cur_seating_lp = log_seating_prob();
		for (size_t extra_step = 0; extra_step < 5; extra_step++) {
			if (!use_expert_labels && infer_alpha_prime) cur_seating_lp = slice_resample_wcrp_param(&log_alpha_prime, cur_seating_lp, -10, 11, .25, log_logalphaprime_prior_density);

			// update gamma
			if (infer_gamma) cur_seating_lp = slice_resample_wcrp_param(&log_gamma, cur_seating_lp, -8, 0, .25, log_loggamma_prior_density);
		}
		
		// update the BKT parameters for each skill
		cout << "  resampling skill parameters" << endl;
		for (size_t extra_step = 0; extra_step < 5; extra_step++) {
			for (boost::unordered_map<size_t, struct bkt_parameters>::iterator table_itr = parameters.begin(); table_itr != parameters.end(); table_itr++) {
				const size_t table_id = table_itr->first;

				// figure out which items are assigned to this skill
				vector<size_t> items_assigned_to_skill;
				for (size_t item = 0; item < num_items; item++) {
					if (seating_arrangement.at(item) == table_id) items_assigned_to_skill.push_back(item);
				}

				// figure out which students would be affected by a change in this skill's BKT parameterization
				vector<size_t> students_to_include, first_exposures;
				for (size_t student = 0; student < num_students; student++) {
					if (train_students.count(student) && studied_any_of(student, items_assigned_to_skill)) {
						students_to_include.push_back(student);
						size_t min_val = item_sequences.at(student).size();
						for (vector<size_t>::const_iterator item_itr = items_assigned_to_skill.begin(); item_itr != items_assigned_to_skill.end(); item_itr++) {
							if (first_encounter.at(student).at(*item_itr) < min_val) min_val = first_encounter.at(student).at(*item_itr);
						}
						first_exposures.push_back(min_val);
					}
				}

				// update the skill's BKT parameters in random order
				vector<double * > param_ptrs;
				param_ptrs.push_back(&(table_itr->second.psi));
				param_ptrs.push_back(&(table_itr->second.mu));
				param_ptrs.push_back(&(table_itr->second.pi1));
				param_ptrs.push_back(&(table_itr->second.pi0));
				generator->shuffle(param_ptrs);
				double cur_ll = skill_log_likelihood(table_id, students_to_include, first_exposures);
				for (vector<double *>::iterator param_itr = param_ptrs.begin(); param_itr != param_ptrs.end(); param_itr++) {
					cur_ll = slice_resample_bkt_parameter(table_id, *param_itr, students_to_include, first_exposures, cur_ll);
				}
			}
		}

		// update the CRP seating arrangement
		if (!use_expert_labels) {
			cout << "  resampling skill assignments" << endl;
			generator->shuffle(all_items);
			for (vector<size_t>::const_iterator item_itr = all_items.begin(); item_itr != all_items.end(); item_itr++) gibbs_resample_skill(*item_itr);
		}
		else cout << "  skipping resampling the skill assignments because we're using the expert labels" << endl;

		clock_t end = clock();
  		double elapsed_ms = (end - begin)/(CLOCKS_PER_SEC/1000.0);
		
		////////////////////////////////////
		size_t train_n, test_n;
		const double train_ll = full_data_log_likelihood(true, train_n);
		const double test_ll = full_data_log_likelihood(false, test_n);

		if (!use_expert_labels) cout << "  log_alpha_prime = " << log_alpha_prime << " (alpha' = " << exp(log_alpha_prime) << ")" << endl;
		cout << "  log_gamma = " << log_gamma << " (gamma = " << exp(log_gamma) << ")" << endl;
		cout << "  TRAINING STUDENTS cross entropy (using single sample) = " << setprecision(5) << (-train_ll / train_n) << endl;
		cout << "  HELDOUT STUDENTS cross entropy (using single sample) = " << setprecision(5) << (-test_ll / test_n) << endl;
		cout << "  iteration completed in " << elapsed_ms / 60000.0 << " minutes." << endl;
		cout << "  current number of skills: " << num_used_skills << endl;
		cout << endl;
		/////////////////////////////////////

		record_sample(replication, test_fold, iter, burn, infer_gamma, elapsed_ms / 60000.0, train_ll, test_ll, train_n, test_n, num_used_skills, log_seating_prob());
		if (dump_skills && iter >= burn) record_skill_assignments(replication, test_fold);
	}

	// write the posterior expectation for each prediction out to file
	for (size_t student = 0; student < num_students; student++) {
		const bool was_heldout = test_students.count(student);
		for (size_t trial = 0; trial < pRT_samples.at(student).size(); trial++) {
			outfile_predictions << replication << " " << test_fold << " " << was_heldout << " " << recall_sequences.at(student).at(trial) << " " << vector_mean(pRT_samples.at(student).at(trial)) << endl;
		}
	}
	
}

#endif
