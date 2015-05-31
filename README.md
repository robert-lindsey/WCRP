# WCRP

Weighted Chinese Restaurant Process model for inferring skill labels in Bayesian Knowledge Tracing

Check out the [paper](http://papers.nips.cc/paper/5554-automatic-discovery-of-cognitive-skills-to-improve-the-prediction-of-student-learning) for more information. 

TODO: high level overview 
TODO: terminology of trial, item, WCRP, ...

## Compiling

WCRP depends on [Boost](http://www.boost.org/) and [GNU GSL](http://www.gnu.org/software/gsl/).

To compile the code, run 

    mkdir build
    cd build
    cmake ..
    make

## Data Format 

WCRP assumes you have your data are in a space-delimited text file with one row per trial. 
The columns should correspond to a student ID, item ID, expert-provided skill id, and whether the student responded correctly. 
The rows for a given student should be sorted by the timestamp (from least to most recent).  

    For example, 
      0 0 0 0
      0 1 0 1
      1 2 1 1
      ...

    denotes that student #0 was initially presented with item #0 and produced an incorrect response, then was presented with item #1 and produced a correct response.
    Student #1 was presented with item #2 and produced a correct response
    According to the human annotator, items #0 and #1 practice skill #0 and item #2 practices skill #1. 

It's important that the student IDs are integers in \[0, 1, ..., # students - 1\], the item IDs are integers in \[0, 1, ..., # items - 1 \], and the expert-provided skills are in \[0, 1, ..., # expert provided skills - 1\]. 



## Usage 

TODO: the following is outdated but will be fixed soon: 


To view the available command line options, type

    ./bin/cross_validation

* The predictions.txt file contains the expected posterior probability of recall for each of the trials of students in a heldout set. It's marginalizing over uncertainty in the skill assignments, number of skills, and Bayesian knowledge tracing parameterizations. See the last few lines of MixtureWCRP::run\_mcmc. There should be one line per student-trial. 
* The skills.txt file has one line per post burn-in period sampling iteration. The goal of the MCMC algorithm is to draw samples from a probability distribution over skill assignments conditioned on the observed student data. Each line in skills.txt is a sample from that distribution. See MixtureWCRP::record\_skill\_assignments.   It's important to note that the skill ids are arbitrary: you can't count on them being the same across samples (i.e., skill 10 on sample 1 may be called skill 42 on sample 2, since the skill ids only denote the partitioning of items into skills given the state of the sampler. The number of skills will vary between samples too.)


Other comments: 
* If you don't have any expert provided skill assignments, you can just use dummy values for the skill id. There's a program option for init\_beta and infer\_beta. If init\_beta is set to 0.0 and you do not set infer\_beta, the code ignores the provided skill values by reverting to a CRP. 
* The skills.txt file is only created if you call the program with the --dump\_skills argument. 

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

