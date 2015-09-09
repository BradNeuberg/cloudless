#!/bin/bash
# Script designed to train on a GPU EC2 instance, copy the results to S3, then potentially
# terminate itself. Note that this script is meant to be run _on_ the EC2 instance itself.
terminate="$1"
if [[ $terminate == "--terminate" ]] ; then
  echo "Terminating EC2 instance after training ends!!"
else
  echo "Note: We are not terminating this EC2 instance after training ends; use --terminate to terminate"
fi
filename=caffe-results-`date +"%m-%d-%y"`-host-`hostname`-time-`date +%s`.tar.gz
./src/cloudless/cloudless.py -p -t -g && \
  echo "Tarring up results..." && \
  tar -cvzf /tmp/`echo $filename` snapshots/ logs/output0003* && \
  echo "Sending results to S3..." && \
  aws s3 cp /tmp/`echo $filename` s3://cloudless-data/ --region us-east-1
if [[ $terminate == "--terminate" ]] ; then
  shutdown -h now
fi

