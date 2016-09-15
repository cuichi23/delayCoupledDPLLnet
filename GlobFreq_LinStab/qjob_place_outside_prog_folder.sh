#!/bin/bash -i 
#$ -S /bin/bash 
# 
# MPI-PKS script for job submission script with ’qsub’. 
# Syntax is Bash with special qsub-instructions that begin with ’#$’. 
# For more detailed documentation, see 
#     https://start.pks.mpg.de/dokuwiki/doku.php/getting-started:queueing_system 
# 

# --- Mandatory qsub arguments 
# Hardware requirements. 
#$ -l h_rss=5000M,h_fsize=2048M,h_cpu=20:00:00,hw=x86_64  

# Specify the parallel environment and the necessary number of slots. 
#$ -pe smp 16

# split stdout and stderr, directory for output files, directory for error files
#$ -j n -o $HOME/Programming/1AqueuePKS/joboutput/ -e $HOME/Programming/1AqueuePKS/joberrors/                                  

# --- Optional qsub arguments 
# Change working directory - your job will be run from the directory 
# that you call qsub in.  So stdout and stderr will end up there. 
#$ -cwd 

# --- Job Execution 
# For faster disk access copy files to /scratch first. 
scratch=/scratch/$USER/$$ 
mkdir -p $scratch 
cd $scratch 
cp -r $HOME/Programming/1AqueuePKS/coupledOscillatorsDPLLqueueBasis/* $scratch

# Execution - running the actual program. 
# [Remember: Don’t read or write to /home from here.] 
echo "Running on $(hostname)" 
echo "We are in $(pwd)" 
# start single case
# python oracle.py ring 3 0.25 0.1 1.45 1.1225198136 0 400
# start many parameter sets, read-out from file, see Array Jobs in documentation
# sed command has to be adjusted, such that it reads out the parameters in the right order from the file
# also add that a parameter-file is written and placed into the results folder
# qsub -t 4-6:2 qjob.sh will spawn 2 jobs, with id 4 and 6
echo python case_bruteforce.py ring 3 `awk -F: 'FNR=='$SGE_TASK_ID' {$12=$10=$9=$8=$7=""; print $0}' $HOME/Programming/1AqueuePKS/DPLLParameters.csv` 1 > "$SGE_TASK_ID".txt
echo python case_bruteforce.py ring 3 `awk 'FNR=='$SGE_TASK_ID' {$12=$10=$9=$8=$7=""; print $0}' $HOME/Programming/1AqueuePKS/DPLLParameters.csv` 1
python case_bruteforce.py ring 3 `awk 'FNR=='$SGE_TASK_ID' {$12=$10=$9=$8=$7=""; print $0}' $HOME/Programming/1AqueuePKS/DPLLParameters.csv` 1

# Finish - Copy files back to your home directory, clean up. 
cp -r $scratch $HOME/Programming/1AqueuePKS/     
cd 
rm -rf $scratch


