# Machine-Reading-Comprehension
2018 AI Challenge: Optional Question Machine Reading Comprehension
Competition：https://challenger.ai/competition/oqmrc2018
Test A: 23rd, Test B: 28th

Some large files cannot upload, such as data source, model checkpoint, results. See baidu pan: https://pan.baidu.com/s/1mVxfs44jxZ6_NSusJce-Jg

## Stage 1：
1. Preprocessing: jieba word segmentation, query choice merging, manual feature generation. See proprecess(2).ipynb.
2. Deep learning models: All refer to papers. Mainly use BIDAF. Tuning history recorded in 记录.docx
* From SQUAD: BIDAF(+attention, +ELMO), HAFN(SLQA), rnet, match-LSTM (all adapted of original version for SQUAD, add shortcut for query to strengthen the query representation)
* For text similarity (because I merge the query with different alternatives, it can be treated as text pairing problem): RCNN(+attention), BIMPM
3. Predict the final results for test set: predict.ipynb
4. Ensemble by averaging best single models: merge.ipynb

Use two word vectors: [Chinese Word Vectors](https://github.com/Embedding/Chinese-Word-Vectors), [fastText](https://fasttext.cc/docs/en/crawl-vectors.html)

Elmo makes worse. Use the [pretrained Chinese Elmo](https://github.com/HIT-SCIR/ELMoForManyLangs). It is too large to generate Elmo for all the dataset. So cut the dataset into many small input txt files, calculate the Elmo features and store in a hdf5 file for each input file, use a generator to read the text and Elmo features. However the IO is still bottleneck. Move feature files to SSD and get acceptable speed.

*Make a big mistake of using validation set and test set when tokenizing, which leads to overfitting, bad performance in test B.*

## Stage 2:
Reorganize the API into py files. Only need to run start.sh.
