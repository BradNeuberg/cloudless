#!/bin/bash
# Script that will rsync over the cloudless/ source to EC2. First argument is the path to your
# EC2 keypair for this EC2 instance, while the second argument is the public DNS name for the EC2
# instance to copy things over to. Note that this script is designed to run _outside_ of EC2 on
# your own machine to copy resources over.
EC2_KEYPAIR="$1"
LOCATION="$2"
cd .. && rsync -rave "ssh -i $EC2_KEYPAIR" -ar \
    --exclude .git/ \
    --exclude snapshots/ \
    --exclude archived_data/ \
    --exclude data/landsat/ \
    --exclude data/planetlab/ \
    --exclude src/annotate/train/static/ \
    --exclude src/cloudless/inference/bbox-regions/ \
    --exclude logs/ \
    --exclude src/caffe_model/bvlc_alexnet/bvlc_alexnet_finetuned.caffemodel \
    cloudless/ ubuntu@$LOCATION:/data/cloudless
