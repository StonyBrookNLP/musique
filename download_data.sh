#!/bin/bash

set -e
set -x

# If gdown doesn't work, you can download files from mentioned URLs manually
# and put them at appropriate locations.
pip install gdown

ZIP_NAME="musique_v0.1.zip"

# URL: https://drive.google.com/file/d/1QC6PRRnIWJ8Z1NBccO04K8CIfZBIdJ8p/view?usp=sharing
gdown --id 1QC6PRRnIWJ8Z1NBccO04K8CIfZBIdJ8p --output $ZIP_NAME
unzip $(basename $ZIP_NAME)
rm $ZIP_NAME

mv musique_v0.1 data

# TODO: prevent these from zipping in.
rm -rf __MACOSX
