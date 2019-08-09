# QRel
Framework for relating questions by their topics
 
This repository features code that was created as part of the NWO project DiscoSumo, conducted by Tilburg University and Radboud University Nijmegen (both in the Netherlands) in collaboration with Sanoma Media BV, with the purpose to return similar and related questions to a given question posted on the Dutch Community Question Answering platform GoeieVraag. The repository includes an API option to be used as a backend service for GoeieVraag.

Check the papers below for more information on methodology and performance:

Kunneman, F., Castro Ferreira, T., Krahmer, E. & Van den Bosch, A. (2019), Question Similarity in Community Question Answering: A Systematic Exploration of Preprocessing Methods and Models, In Proceedings of the International Conference Recent Advances in Natural Language Processing (pp. ...)

For other questions, please contact us via f.kunneman@gmail.com or thiago.castro.ferreira@gmail.com


Installation:

python setup.py install
python -m spacy download nl_core_news_sm


Data:

The system only functions if the correct files are included in the repository, which are listed in qrel/modules/relate.py
These files are shared separately, due to their size and privacy restrictions. 


Usage:

- Test

Provided that the data is stored in the right location, the question similarity and question relatedness functions of the system can be tested by running the following command in commandline (when located in the root of this repository):

python qrel/modules/relate.py test

To test the addition of several questions from a file, run the following command:

python qrel/modules/relate.py test_many

- API

To use the system as an api-service, first run the following command:

python qrel/api.py

A local service is now initiated, that will respond to relatedness requests like the following (in the command line):

curl -i -X GET -H "Content-Type: application/json" -d '{"text":"nintendo switch of playstation 4?", "id":"5678910"}' http://localhost:5000/related

The service will return json-formatted output, with the following fields:
	"questiontext"	: the text of the search question
	"qid"			: the id of the search question
	"related"		: a selection of 5 related questions, as a list of lists with 1) the id of the related question 2) the text of the related question 3) the similarity score 4) the topic to which this question most relates
	"candidates"	: list of question ids that were found possibly related candidates, could be used to update their relatedness information

The API could also be ran to retrieve the most similar questions to a given question:

curl -i -X GET -H "Content-Type: application/json" -d '{"text":"nintendo switch of playstation 4?", "model":"ensemble"}' http://localhost:5000/similar

The service will return json-formatted output, with the following fields:
	"questiontext"	: the text of the search question
	"similar"		: the 5 most similar questions, as a list of lists with 1) the id of the similar question 2) the text of the similar question 3) the similarity score 4) an assessment if it is completely similar ('0' for similar, '0' for not similar); this only applies to the ensemble model, any of the other models (bm25, trlm, softcosine) always return '0'

- Add many questions

To update the dataset with a new file with many questions, run the following command:

python qrel/modules/relate.py [path_to_file_with_new_questions.json]

The file should be json-formatted, as a list of dictionaries (representing the questions) which should at least contain the following fields:
	"id"			: the question id
	"questiontext"	: the text of the question

The other fields they could optionally include are:
	"tokens"	: the word tokens of the question text
	"lemmas"	: normalized versions of these word tokens
	"pos"		: the grammatical categories of these word tokens
	"topics"	: the words and phrases that in the question that reflect topics
	"related"	: the id's and texts of the questions that are selected as related

The questions in this file will be added to the original questions and saved. If the optional fields are not included, the system will extract them. 

- update relatedness



