This project was written on Windows 8 and upon completion tested by a reviewer on __
The Python version used on both operating systems was 3.7.3.

## Name
main.py - a program managing a collection of author profiles, based on which classification for texts of (supposedly) unknown authorship can be performed

## Description
Given a set of candidate authors, text samples of known authorship covering all the candidate authors(= training corpus)
and text samples of (supposedly) unknown authorship (= test corpus), authorship attribution describes the problem of deducing
an author proﬁle from the former to be used in attributing a candidate author to each of the later (Stamatatos, 2009).
The first step of deducing profiles is done by the class *AuthorModel* and the second step of attributing a candidate author
based on these profiles is done by the class *AuthorIdent*. *AuthorModel* is a subclass of *dict* and its instances can be understood as feature vectors.
The **nltk** library was used to perform the natural language processing necessary for the subsequent feature extraction and mainly visible in the 
function *AuthorModel._nlp* where tokens are enriched with e.g. information about their POS-Tag and lemma, before they are finally returned as **namedtuple**s 
in blocks of complete sentences. Feature extration and normalization is performed in the Function *AuthorModel.__extract_features*.
The User interface to those methods is the function *AuthorModel.train*. Furthermore the functions *AuthorModel.write_csv* and *AuthorModel.read_csv* offer
the possibility of respectively saving and loading trained feature vectors.  
The class *AuthorIdent* can be seen as a feature vector or profile management system, providing several services to the user including adding (*AuthorIdent.train*)
and deleting (*AuthorIdent.forget*) profiles as well as performing the classification task (*AuthorIdent.classify*) based on the added profiles.
While those two classes are the main pillars of the project, another module *mtld.py* includes an implementation of the vocabulary richness score MTLD chosen because
it is said to be the measure most immune to varying text lengths a characteristic deemed important for the chosen data set.
*distributions.py* includes subclasses of **Counter**. They are used for normalizing counts and have been used to produce visualizations of the data
with **matplotlib** and **numpy**, but this functionality is not directly accessible in the framework of the project.

Other modules used are **tqdm** and from the standard library **functools**, **logging**, **os**, **sys**, **types**.

## Requirements
The following steps are necessary for initializing the project environment:
+ Create and activate Virtual Environment (recommended)
  On Windows:
   ```sh
   $ python -m venv env
   $ env\Scripts\activate
   ```
   On Linux:
    ```sh
   $ python3 -m venv env
   $ env/bin/activate
   ```
+ Install dependencies (replace *python* with *python3* on Linux):
  ```sh
   $ python -m pip install -r requirements.txt
   ```
+ Download NLTK data (replace *python* with *python3* on Linux):
  ```sh
   $ python -m nltk.downloader punkt
   $ python -m nltk.downloader averaged_perceptron_tagger
   $ python -m nltk.downloader wordnet
   ```
+ Download the [Gutenberg Dataset](https://web.eecs.umich.edu/~lahiri/gutenberg_dataset.html) .
+ Unzip *Gutenberg.zip*.
+ Execute the following command from the root directory (**execution time: ~15min**)(replace *python* with *python3* on Linux):
  ```sh
   $ python scripts/split_data.py PATH_TO_UNZIPPED_GUTENBERG
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
   python main.py --preprocess FILENAME
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
   + Path to a csv-file containing lines of the form <author>\t<pretrained model csv-filepath> .
   + It is advisable and necessary that catalogs are always accessed from the same working directory.
Otherwise problems involving the relative paths to the profiles saved in it will occur.
+ FILENAME
    + Utf-8 encoded raw txt-file.
+ SOURCE
    + Utf-8 encoded preprocessed txt-file where tokens are separated with whitespaces and each sentence is on its separate line.

## Examples
+ `python main.py --catalog data/gutenbergident.txt --train  "Anthony Trollope" "corpus/training/Anthony Trollope"`
+ `python main.py --catalog data/gutenbergident.txt --forget "Anthony Trollope"`
+ `python main.py --catalog data/gutenbergident.txt --classify "corpus/test/Anthony Trollope/Anthony Trollope___A Ride Across Palestine"`

## Author
Wencke Liermann  
Uni Potsdam  
Computerlinguistik 4. Semester  
SoSe20  

For further inquiries and feedback, please contact: wliermann@uni-potsdam.de
