#!/bin/bash
# This script is going to create a zip file by
#
# 1) creating a tmp/ directory
#
# 2) copying in tmp/ the following files/directories:
#
# competition.yaml
# *.jpg
# *.png
# *.gif
# *.html
# assets/
#
# 3) zipping and moving to tmp/ the contents or the following dirs
# public_data/
# reference_data/
# starting_kit/
# scoring_program/
# starting_kit/

# Clear last output
#echo 'Cleaning output files of last local execution...'
#ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
#echo 'Using ROOT_DIR: '$ROOT_DIR

#STARTING_KIT_DIR=$ROOT_DIR/../starting_kit/
#cd $STARTING_KIT_DIR
#rm -rf scoring_output
#rm -rf sample_result_submission
#ls | grep checkpoints_* | xargs rm -rf
#cd $ROOT_DIR

DATE=`date "+%Y-%m-%d-%H-%M-%S"`
DIR='tmp/'
# Delete $DIR if exists
rm -rf $DIR
mkdir $DIR
cp '../'*.jpg $DIR
cp '../'*.png $DIR
cp '../'*.gif $DIR
cp '../'*.html $DIR
cp '../'*.yaml $DIR
cp -r '../assets' $DIR
cd .. # bundle/

# Begin zipping each folder
for filename in input_data reference_data scoring_program ingestion_program starting_kit
do
  cd $filename
  echo 'Zipping: '$filename
  zip -q -r "../utilities/"$DIR$filename .;
  cd .. # bundle/
done

# Zip all to make a competition bundle
cd "utilities/"$DIR
zip -q -r "../bundle_"$DATE .;
