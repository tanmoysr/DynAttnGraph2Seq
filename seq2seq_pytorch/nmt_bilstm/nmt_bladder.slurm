#!/bin/sh

## General comments about SLURM batch scripts
## - Lines begining with "#SBATCH" give info to Slurm, but ignored otherwise
## - An SBATCH command is "commented out" (ignored) if line begins with 2 "#"
## - If a command is repeated with different values, later overrides earlier
## - Try to avoid using this override feature.  It may cause confusion.
## - Anything in <angle brackets> needs to be modified by you (remove <> too)

## Slurm terminology:
##  Partition - A Job queue, submitted jobs stay here until resources are free
##  Node      - A computer in the cluster (e.g. NODE020)
##  Core      - This is essentially a CPU in a node
##  Task      - A standard heavy-weight process (i.e. not a thread)
##  Array Job - A job that runs multiple sub-jobs (e.g. for parameter sweeps)

## Specify the name for your job, this is the job name by which Slurm will
## refer to your job.  This can be different from the name of your executable
## or the name of your script file. But it must be the same like your slurm file.
## Slurm file name should be like job-name.
#SBATCH --job-name=nmt_bladder

## General partitions: all-LoPri, all-HiPri, bigmem-LoPri, bigmem-HiPri, gpuq
##    all-*     Will run jobs on (almost) any node available
##    bigmem-*  Will run jobs only on nodes with 512GB memory
##    *-HiPri   Will run jobs for up to 12 hours
##    *-LoPri   Will run jobs for up to 5 days
##    gpuq      Will run jobs only on nodes with GPUs (40, 50, 55, 56)
## Restricted partitions: CDS_q, CS_q, STATS_q, HH_q, GA_q, ES_q, COS_q
##                        Provide high priority access for contributors
#SBATCH --partition=all-HiPri

## Deal with output and errors.  Separate into 2 files (not the default).
## May help to put your result files in a directory: e.g. /scratch/%u/logs/...
## NOTE: %u=userID, %x=jobName, %N=nodeID, %j=jobID, %A=arrayID, %a=arrayTaskID
#SBATCH --output=/scratch/%u/%x-%N-%j.out  # Output file
#SBATCH --error=/scratch/%u/%x-%N-%j.err   # Error file
#SBATCH --mail-type=BEGIN,END,FAIL         # ALL,NONE,BEGIN,END,FAIL,REQUEUE,..
#SBATCH --mail-user=tchowdh6@gmu.edu     # Put your GMU email address here

## You can improve your scheduling priority by specifying upper limits on
## needed resources, but jobs that exceed these values will be terminated.
## Check your "Job Ended" emails for actual resource usage info as a guide.
#SBATCH --mem=128G        # Total memory needed per task (units: K,M,G,T)
#SBATCH --time=0-50:00  # Total time needed for job: Days-Hours:Minutes

## ##### Optional stuff #####
## From this point on, Slurm SBATCH commands are commented out (ignored).
## If you want to use any of them, remove one of the '#' before the SBATCH.
## If there is a similar command above this point, it is a good idea to
## comment that out as well to avoid confusion.

## ----- GPU Jobs -----
## Jobs that use one or more GPUs need to both run on the gpuq partition
## and to request gpus using "--gres=gpu:<G>".
##SBATCH --partition=gpuq
##SBATCH --gres=gpu:2       # Number of GPUs needed
##SBATCH --nodelist=NODE0<XX> # If you want to run on a specific node

## ----- Parallel Threads -----
## Some programs and libraries (OpenMP) implement parallelism using threads
## (light-weight sub-processes). Advantages: Less processing overhead and
## ability to share memory.  Disadvantages: All threads must run on the same
## node.  Make sure that the resources you request are feasible,
## e.g. --cpus-per-task must be <= # of cores on a node.
#SBATCH --cpus-per-task=5   # Request extra CPUs for threads
##SBATCH --mem-per-cpu <M>     # If your threads us a lot of memory, and you
                               # plan to vary # of threads, use this not --mem=

## ----- Parallel Processes  -----
## Some libraries (MPI) implement parallelism using processes that communicate.
## This allows tasks to run on any set of cores in the cluster.  Programs can
## use this approach in combination with threads (if designed to).
## In that case, use parameters from here and the "Parallel Threads" section.
##SBATCH --ntasks <T>          # Number of processes you plan to launch
##SBATCH --nodes <N>           # If you want some control over how tasks are
                               #    distributed on nodes.  <T> >= <N>
##SBATCH --ntasks-per-node <Z> # If you want more control over how tasks are
                               #    distributed on nodes.  <T> = <N> * <Z>

## ----- Array Jobs -----
## Array jobs run multiple similar sub-jobs (e.g. for parameter sweeps).
## Programs can find out which sub-job they belong to by checking
## the $SLURM_ARRAY_TASK_ID environment variable.
##SBATCH --array=2,7,10       # List of arrayTaskIDs -or-
##SBATCH --array=1-12:2%3     # Start-End[:Increment][%MaxAtOnce] []=optional
##SBATCH --output=/scratch/%u/%x-%N-%A-%a.out  # Output file
##SBATCH --error=/scratch/%u/%x-%N-%A-%a.err   # Error file

## Load the relevant modules needed for the job
module load python/3.8.4
source tc-site-virtualenv-3.8.4-no-sys-pack/bin/activate
## Run your program or script
cd ../../scratch/tchowdh6/NMT_bladder/
python run_nmt_bilstm.py  --train
python run_nmt_bilstm.py  --test
## Unload module
## module purge

