conda config --add channels conda-forge
conda install zip
rm -f 2022_24790_coding.zip #change here to your student id
zip -r 2022_24790_coding.zip ./skipgram/utils/*.py ./skipgram/*.py ./skipgram/*.png ./skipgram/saved_params_20000.npy ./skipgram/saved_state_20000.pickle ./sentimentAnalysis/*.ipynb
#change above to your student id
                   