#!/bin/bash
# Script designed to train on a GPU EC2 instance, copy the results to S3, then potentially
# terminate itself. Note that this script is meant to be run _on_ the EC2 instance itself.
# The first argument is the log number to use when generating log output, the second argument
# is a note that will be output to the log to identify any hyperparameters used in this training
# run, while --terminate is an optional argument that will terminate the EC2 instance when it is
# finished training to save money. Note that these arguments are all position dependent.
# Example: ./src/aws_scripts/train.sh --log_num 1 --note "Important note" --terminate

set -e
set -o pipefail

S3_REGION=us-east-1

log_num="$2"
note="$4"
terminate="$5"

if [ -z "$log_num" ]; then
  echo "You must provide a log number for saving output"
  exit 1
fi

if [ -z "$note" ]; then
  echo "You must provide a note with training details"
  exit 1
fi

terminate_command=""
if [[ $terminate == "--terminate" ]]; then
  echo "Terminating EC2 instance after training ends!"
  terminate_command="shutdown -h now"
else
  echo "Note: We are not terminating this EC2 instance after training ends; use --terminate to terminate"
fi

filename=caffe-results-`date +"%m-%d-%y"`-host-`hostname`-time-`date +%s`.tar.gz

mkdir -p logs/
mkdir -p snapshots/

./src/cloudless/train/train.py --log_num $log_num --note "$note" && \
  echo "Tarring up results..." && \
  tar -cvzf /tmp/`echo $filename` snapshots/ logs/* && \
  echo "Sending results to S3..." && \
  aws s3 cp /tmp/`echo $filename` s3://cloudless-data/ --region $S3_REGION && \
  $terminate_command
