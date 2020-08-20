This project was written on Windows 8 and upon completion tested by a reviewer on __
The Python version used on both operating systems was 3.7.3.

## Name
main.py - a program managing a collection of author profiles, based on which classification for texts of (supposedly) unknown authorship can be performed

## Description
Given a set of candidate authors, text samples of known authorship covering all the candidate authors(=training corpus)
and text samples of (supposedly) unknown authorship (=test corpus), authorship attribution describes the problem of deducing
an author proﬁle from the former to be used in attributing a candidate author to each of the later (Stamatatos, 2009).
The first step of deducing profiles is done by the class AuthorIdent and the second step of attributing a candidate author
based on these profiles is done by the class AuthorModel. Those two classes are the main pillars of the project.

## Requirements
The following steps are necessary for initializing the project environment:
+  Create and activate Virtual Environment (recommended)
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
+ Install dependencies
  ```sh
   $ pip install -r requirements.txt
   ```
+ Download the [Gutenberg Dataset](https://web.eecs.umich.edu/~lahiri/gutenberg_dataset.html) .
+ Unzip *Gutenberg.zip* to the root directory of the project.
+ Execute the script .... bla from the root directory (**execution time: ~15min**).
+ 

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
Wencke Liermann\n
Uni Potsdam\n
Computerlinguistik 4. Semester\n
SoSe20\n

For further inquiries and feedback, please contact: wliermann@uni-potsdam.de
