# WCRP

UNDER CONSTRUCTION. 


Weighted Chinese Restaurant Process model for inferring skill labels in Bayesian Knowledge Tracing

Check out the [paper](http://papers.nips.cc/paper/5554-automatic-discovery-of-cognitive-skills-to-improve-the-prediction-of-student-learning) for more information. 

TODO: high level overview , motivation 
TODO: terminology of trial, item, WCRP, ...


## Compiling

WCRP is written in C++. It depends on the [Boost](http://www.boost.org/) and [GNU GSL](http://www.gnu.org/software/gsl/) libraries. 

After installing Boost and GNU GSL, run

    mkdir build
    cd build
    cmake ..
    make

to compile the code. 

## Data Format 

WCRP assumes that your data are in a space-delimited text file with one row per trial. 
The columns should correspond to a trial's student ID, item ID, expert-provided skill ID, and whether the student produced a correct response in the trial. 
The rows for a given student should be ordered by the timestamp (from least to most recent).  

It's important that the student IDs are integers in \[0, ..., S - 1\], the item IDs are integers in \[0, ..., I - 1 \], and the expert-provided skills are in \[0, ..., E - 1\] where S is the number of students in your dataset, I is the number of items, and E is the number of expert-provided skills. 
For example, your data should look like the following: 

    0 0 0 0
    0 1 0 1
    1 2 1 1
    ...

In the above example, student #0 was initially presented with item #0 and produced an incorrect response, and then was presented with item #1 and produced a correct response.
Student #1 was presented with item #2 and produced a correct response
According to the human annotator, items #0 and #1 practice skill #0 and item #2 practices skill #1. 


## Usage 

TODO: the following is outdated but will be fixed soon: 


#### Finding the most likely skill assignments

The command

    ./bin/find_skills --map_estimate --datafile dataset.txt --savefile map_skills.txt --num_iterations 3000 --burn 1000

will run the Gibbs sampler on the data in dataset.txt for 3000 iterations and discard the first 1000 iterations as burn-in. It'll then save the maximum a posteriori (MAP) estimate of the skill assignments to the file map_skills.txt. The ith number in map_skills.txt is the skill ID of item i. 


#### Determining the distribution over skill assignments


The command

    ./bin/find_skills --datafile dataset.txt --savefile sampled_skills.txt --num_iterations 3000 --burn 1000

will run the Gibbs sampler on the data in dataset.txt for 3000 iterations and discard the first 1000 iterations as burn-in. 
It'll produce the text file sampled_skills.txt which will 2000 lines (one per post burn-in iteration). 
The goal of the MCMC algorithm is to draw samples from a probability distribution over skill assignments conditioned on the observed student data. 
Each line in skills.txt is a sample from that distribution.

It's important to note that the skill ids are arbitrary: you can't count on them being the same across samples (i.e., skill 10 on sample 1 may be called skill 42 on sample 2, since the skill IDs only denote the partitioning of items into skills given the state of the sampler. The number of skills will vary between samples too.)


#### Cross Validation Simulations 

TODO: description of split files 

To view the available command line options, type

    ./bin/cross_validation


The resulting predictions.txt file contains the expected posterior probability of recall for each of the trials of students in a heldout set. It's marginalizing over uncertainty in the skill assignments, number of skills, and Bayesian knowledge tracing parameterizations. See the last few lines of MixtureWCRP::run\_mcmc. There should be one line per student-trial. 

If you don't have any expert provided skill assignments, you can just use dummy values for the skill id. There's a program option for init\_beta and infer\_beta. If init\_beta is set to 0.0 and you do not set infer\_beta, the code ignores the provided skill values by reverting to a CRP. 



## Implementation Notes



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

