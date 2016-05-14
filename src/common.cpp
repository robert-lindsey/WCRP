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

#ifndef COMMON_CPP
#define COMMON_CPP

#include "common.hpp"

// reads a tab delimited file with the columns: student id, item id, skill id, recall success
// all ids are assumed to start at 0 and be contiguous
void load_student_data(const char * filename, std::vector< std::vector<bool> > & recall_sequences, std::vector< std::vector<size_t> > & item_sequences, size_t & num_students, size_t & num_items, size_t & num_skills) {

    num_students=0, num_items=0;
    size_t student, item, recall;

    std::ifstream in(filename);
    if (!in.is_open()) {
        std::cerr << "couldn't open " << std::string(filename) << std::endl;
        exit(EXIT_FAILURE);
    }

    // figure out how many students, items, and skills there are
    while (in >> student >> item >> recall) {
        num_students = std::max(student+1, num_students);
        num_items = std::max(item+1, num_items);
    }
    in.close();

    std::cout << std::string(filename) << " has " << num_students << " students and " << num_items << " items" << std::endl;

    // initialize
    recall_sequences.resize(num_students);
    item_sequences.resize(num_students);

    // read the dataset
    in.open(filename);
    while (in >> student >> item >> recall) {
        recall_sequences[student].push_back(recall);
        item_sequences[student].push_back(item);
    }
    
    in.close();
}


// reads a text file with expert-provided skill ids
void load_expert_labels(const char * filename, std::vector<size_t> & provided_skill_labels, const size_t num_items) {

    std::ifstream in(filename);
    if (!in.is_open()) {
        std::cerr << "couldn't open " << std::string(filename) << std::endl;
        exit(EXIT_FAILURE);
    }

    size_t skill;
    size_t num_lines = 0;
    while (in >> skill) provided_skill_labels[num_lines++] = skill;
    in.close();
    assert(num_lines == num_items);
}


// reads the K-fold cross validation assignments
void load_splits(const char * filename, std::vector<std::vector<size_t> > & fold_nums, size_t & num_folds, const size_t num_students) {

    std::ifstream in(filename);
    if (!in.is_open()) {
        std::cerr << "couldn't open " << std::string(filename) << std::endl;
        exit(EXIT_FAILURE);
    }

    num_folds = 0;
    while(!in.eof()) {
        // read a line
        std::string line;
        getline(in, line);
        boost::trim(line);
        if (line.empty()) break;

        // split on whitespace
        std::vector<std::string> fields;
        boost::split(fields, line, boost::is_any_of(" \t"));
        assert(fields.size() == num_students);

        std::vector<size_t> replication_fold_nums(fields.size());
        for (size_t student = 0; student < fields.size(); student++) {
            replication_fold_nums[student] = boost::lexical_cast<size_t>(fields[student]);
            num_folds = std::max(replication_fold_nums[student]+1, num_folds);
        }
        fold_nums.push_back(replication_fold_nums);
    }

    std::cout << "# replications to run = " << fold_nums.size() << std::endl;
    std::cout << "# folds per replication = " << num_folds << std::endl;
}

#endif
