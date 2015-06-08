# WCRP

WCRP is a Weighted Chinese Restaurant Process model for inferring skill labels in Bayesian Knowledge Tracing. 

Check out the [paper](http://papers.nips.cc/paper/5554-automatic-discovery-of-cognitive-skills-to-improve-the-prediction-of-student-learning) for more information. 


## Compiling

WCRP is written in C++. It depends on the [Boost](http://www.boost.org/) and [GNU GSL](http://www.gnu.org/software/gsl/) libraries. 

After installing Boost and GNU GSL, run

    mkdir build
    cd build
    cmake ..
    make

to compile the code. 


## Data Format 

#### (Required) Student Responses

WCRP assumes that your student data are in a space-delimited text file with one row per trial. 
The columns should correspond to a trial's student ID, item ID, and whether the student produced a correct response in the trial. 
The IDs should be integers beginning at 0, and the rows for each student should be ordered from least to most recent. 

For example, your data should look like the following: 

    0 0 0
    0 1 1
    1 2 1
    
In the above example, student 0 was initially presented with item 0 and produced an incorrect response, and then was presented with item 1 and produced a correct response; 
student 1 was presented with item 2 and produced a correct response. 


#### (Optional) Expert-provided skills  

Our model's nonparametric prior distribution over skill labels can leverage skill labels provided by a human domain expert. 
If you want to provide them to our model, create a text file with one line per item. 
The ith line should be the expert-provided skill ID for the ith item. 

The parameter beta in the model controls how much the prior is drawn toward the expert-provided skills.
A value of 0 will have the model ignore the expert-provided skills, and the model will deterministically use
the expert-provided skills as beta approaches 1. 
The command line options of WCRP allow you to hold beta constant at a specified value or to have the model give
beta the Bayesian treatment by treating it as another random nuisance variable. 


## Usage 

Compiling WCRP produces two executable files: find_skills and cross_validation. 
Each has a variety of command line options you can view via the command line argument --help. 


#### Finding the most likely skill assignments

The command

    ./bin/find_skills --datafile dataset.txt --savefile map_estimate_skills.txt --map_estimate 

will run the Gibbs sampler on the data in dataset.txt using default settings. It'll then save the maximum a posteriori (MAP) estimate of the skill assignments to the file map_estimate_skills.txt. The ith entry in map_skills.txt is the skill ID of item i. 


#### Sampling the posterior distribution over skill assignments 

The command

    ./bin/find_skills --datafile dataset.txt --savefile sampled_skills.txt --iterations 3000 --burn 1000

will run the MCMC algorithm on the data in dataset.txt for 3000 iterations and discard the first 1000 iterations as burn-in. 
It'll produce the text file sampled_skills.txt which will have 2000 lines (one per post burn-in iteration). 
The goal of the MCMC algorithm is to draw samples from a probability distribution over skill assignments conditioned on the observed student data. 
Each line in skills.txt is a sample from that distribution.

The skill IDs are sample-specific: you can't count on them being the same across samples because they only denote the partitioning of items into skills given the state of the Markov chain. 
The number of skills will typically vary between samples too.


#### Running cross validation simulations on heldout students 

[THESE INSTRUCTIONS WILL BE UPDATED SOON]

The executable cross_validation runs K-fold cross validation on your dataset.
It requires a space-delimited text file ("foldfile") with one row per cross validation simulation you want to run.
There should be one column per student in your dataset. 
Each entry indicates the fold number of the student in that replication. For example, 

    0 0 1 1 2 2
    0 1 2 0 1 2

denotes that in the first replication, students 0 and 1 are in fold 0, students 2 and 3 are in fold 1, and students 4 and 5 are in fold 2. The second replication has students 0 and 3 in fold 0, students 1 and 4 in fold 1, and students 2 and 5 in fold 3. 

The command

    ./bin/cross_validation --datafile dataset.txt --foldfile folds.txt --predfile predictions.txt 

will produce the text file predictions.txt containing the expected posterior probability of recall for each of the trials of the students in the heldout set. 
There will be one line per replication-fold-student-trial. 


## License and Citation

This code is released under the [MIT License](https://github.com/robert-lindsey/WCRP/blob/master/LICENSE.md).

Please cite our paper in your publications if it helps your research: 

    @incollection{lindsey2014,
      title = {Automatic Discovery of Cognitive Skills to Improve the Prediction of Student Learning},
      author = {Lindsey, Robert V and Khajah, Mohammad and Mozer, Michael C},
      booktitle = {Advances in Neural Information Processing Systems 27},
      editor = {Z. Ghahramani and M. Welling and C. Cortes and N.D. Lawrence and K.Q. Weinberger},
      pages = {1386--1394},
      year = {2014},
      publisher = {Curran Associates, Inc.},
    }

