This project was written on Windows 8 and upon completion tested by a reviewer on Windows 10.
The Python version used on both operating systems was 3.7.3.

## Name
main.py - a program managing a collection of author profiles, based on which classification of texts of (supposedly) unknown authorship can be performed

## Description
Given a set of candidate authors, text samples of known authorship covering all the candidate authors (= training corpus)
and text samples of (supposedly) unknown authorship (= test corpus), authorship attribution describes the problem of deducing
an author proﬁle from the former to be used in attributing a candidate author to each of the latter (Stamatatos, 2009).
The first step of deducing profiles is done by the class *AuthorModel* and the second step of attributing a candidate author
based on these profiles is done by the class *AuthorIdent*. The instances of *AuthorModel* can be understood as feature matrices,
its attributes are feature vectors for features belonging to different categories.
The **nltk** library was used to perform the natural language processing necessary for the underlying feature extraction and mainly visible in the 
function *AuthorModel._nlp* where tokens are enriched with e.g. information about their POS-Tag and lemma, before they are finally returned as **namedtuple**s 
in blocks of complete sentences. Feature extraction is performed in the Function *AuthorModel._extract_features*.
To return all normalized features in a single vector the function *AuthorModel.normalized_feature_vector* is used.
The user interface to those methods is the function *AuthorModel.train*. Furthermore, the functions *AuthorModel.write_json* and *AuthorModel.read_json* offer
the possibility of respectively saving and loading trained feature matrices.  
The class *AuthorIdent* can be seen as a feature matrix or profile management system, providing several services to the user including adding (*AuthorIdent.train*)
and deleting (*AuthorIdent.forget*) profiles as well as performing the classification task (*AuthorIdent.classify*) based on the added profiles.
While those two classes are the main pillars of the project, another module *mtld.py* includes an implementation of the vocabulary richness score MTLD chosen because
it is said to be the measure most immune to varying text lengths a characteristic deemed important for the chosen data set.
*distributions.py* includes subclasses of **MutableMapping**. They are used for normalizing counts and have been used to produce visualizations of the data
with **matplotlib** and **numpy**, but this functionality is not directly accessible in the framework of the project.

Other modules used are **tqdm** and from the standard library **functools**, **logging**, **os**, **sys**, **types**, **unittest**, **re**, **json**.

## Requirements
Depending on the operating system *python* may have to be replaced with *py* or *python3*.
The following steps are necessary for initializing the project environment:
+ (recommended) Create and activate Virtual Environment 
  On Windows:
   ```sh
   $ python -m venv env
   $ env\Scripts\activate
   ```
   On Linux:
    ```sh
   $ python3 -m venv env
   $ source ./env/bin/activate
   ```
+ Install dependencies:
  ```sh
   $ python -m pip install -r requirements.txt
   ```
+ Download NLTK data:
  ```sh
   $ python -m nltk.downloader punkt
   $ python -m nltk.downloader averaged_perceptron_tagger
   $ python -m nltk.downloader wordnet
   ```
+ Download the [Gutenberg Dataset](https://web.eecs.umich.edu/~lahiri/gutenberg_dataset.html) .
+ Unzip *Gutenberg.zip*.
+ (recommended) If you'd like to choose your own author set, modify *data\author_config.json*. Otherwise material from 10 predefined authors will be selected.
+ Execute the following command from the root directory of the project(**execution time: ~15min**):
  ```sh
   $ python scripts\split_corpus.py PATH_TO_UNZIPPED_GUTENBERG
   ```
   If saved in the root directory of the project e.g.:
  ```sh
   $ python scripts\split_corpus.py Gutenberg
   ```

## Synopsis
Enter a command following the scheme below in order to:
+ Add a new class to the classifier:
    ```sh
    python main.py --catalog CATALOG --train AUTHOR SOURCE
    ```
+ Delete an existing class from the classifier:
    ```sh
    python main.py --catalog CATALOG --forget AUTHOR
    ```
+ Delete the catalog and every class inside it:
    ```sh
    python main.py --catalog CATALOG --destroy
    ```
+ Perform the classification task based on a catalog with trained classes:
    ```sh
    python main.py --catalog CATALOG --classify SOURCE
    ```
+ Preprocess a file to make it admittable as a SOURCE argument:
    ```sh
    python main.py --preprocess FILENAME GOAL
    ```
+ Run all unittests.
    ```sh
    python main.py --test
    ```
Additionally a target --verbosity can be used with any of the above schemes to adjust the amout of output (0=errors, 1=warnings and above, 2=info and above). Default is 1.

## Arguments
+ AUTHOR
    + Name of the class.
    + This name will be put out, when the classifier identifies a text as belonging to the class.
+ CATALOG
   + Path to a csv-file containing lines of the form <author>\t<pretrained model JSON-filepath> created by AuthorIdent.
   + It is advisable and necessary that catalogs are always accessed from the same working directory.
Otherwise problems involving the relative paths to the profiles saved in it will occur.
+ FILENAME
    + Utf-8 encoded raw txt-file.
    + Preferable larger amounts of text, but at least containing two sentences, three consecutive words and one punctuation mark.
+ SOURCE
    + Utf-8 encoded preprocessed txt-file where tokens are separated with whitespaces and each sentence is on its separate line.
+ GOAL
    + Path to save the resulting preprocessed file at.

## Examples
For those commands to be succesfully executed the passed file paths of course have to exist.
+ `python main.py --catalog data\gutenbergident.csv --forget "Anthony Trollope"`
+ `python main.py --catalog data\gutenbergident.csv --train  "Anthony Trollope" "corpus\\training\\Anthony Trollope"`
+ `python main.py --catalog data\gutenbergident.csv --classify "corpus\\test\\Anthony Trollope\\Anthony Trollope___Lady Anna.txt"`
+ `python main.py --preprocess "Aldous Huxley___Mortal Coils.txt" "Aldous Huxley___Mortal Coils.pre"`


## Accuracy
+ Executing the following command from the root directory of the project will calculate the accuracy over the test set (**execution time: ~30min**):
  ```sh
   $ python scripts\evaluate.py CATALOG FILENAME TEST_DIRECTORY
   ```
   If saved in the root directory of the project e.g.:
  ```sh
   $ python scripts\evaluate.py data\gutenbergident.csv data\eval.csv corpus\test
   ```

## Running Time
Training for all the 10 chosen authors takes up 1h 30min.
I chose the authors with the largest amount of data under their name to have as much data in the test set as possible.
But for anyone testing the program I would recommend specifying only up to three authors in *data\author_config.json*.
For example Alexander Pope and Abraham Lincoln.

## Author
Wencke Liermann  
Uni Potsdam  
Computerlinguistik 4. Semester  
SoSe20  
  
For further inquiries and feedback, please contact: wliermann@uni-potsdam.de
