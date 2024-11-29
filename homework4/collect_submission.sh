conda config --add channels conda-forge
conda install zip
rm -f 2022_24790_coding.zip #change here to your student id
zip -r 2022_24790_coding.zip ./q1/*.ipynb ./q2/*.py ./q2/*.log
#change above to your student id