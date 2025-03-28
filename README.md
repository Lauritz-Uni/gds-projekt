
# Downloading the project
Run in a terminal window:
```
git clone https://github.com/Lauritz-Uni/gds-projekt.git
cd gds-projekt
```

If you are using VSCode, then you can simply run
```
code .
```
to open the project in VSCode

# Preprocessor
The preprocessor is written in rust and therefore takes some more time to get a running version.
## How to build
First, [dowload rust and cargo](https://www.rust-lang.org/tools/install). There might be additional dependicies when installing rust (such as Visual Studio Build Tools for Windows). The rust installation will exaplain where to download these additional dependencies.

To compile preprocessor, run this in a terminal window in the rust-preprocess folder
```
cargo build --release
```
This will take about 1-2 mins to compile.


## How to run
You can run the python file "preprocess.py" to do all preprocessing steps. This assumes you have both liar data set and 955,000_rows dataset in the data directory with default names.

### Manually run the preprocessor

It is assumed you are running these commands in the root folder of the project (probably named "gds-projekt").

You can run 
```
./rust-preprocess/target/release/rust-preprocess.exe --help
```
to get an overview of all the arguments available.

To preprocess a file with only the different splits run this in a terminal window
```
./rust-preprocess/target/release/rust-preprocess.exe --input "./data/995,000_rows.csv" --output "./output/995,000_rows_processed.csv"
```
This will create three files in the output folder. The name "995,000_rows_processed" will define what will be prepended to the splits.
Note: this specific command creates about 2-3 gb of additional data.

It is possible to get the entirety of the processed file (including the output of cleaning steps) by running:
```
./rust-preprocess/target/release/rust-preprocess.exe --input "./data/995,000_rows.csv" --output "./output/995,000_rows_processed.csv" --keep-processed
```
NOTE: This will create a rather large file that is about 10 gb in addition to aforementioned splits. Only run the above command if you need the entire preprocessed file.
# Models
Our models are written in Python 3.12.x. It is assumed that you have any Python 3.12.x installed and that you are using it to run the files.

## Virtual enviroment initialization
You will probably want to create a virtual enviroment. We used Conda while coding this project, but most other types of virtual enviroments should work.

If you are using VSCode, you can do this by opening the command pallet and searching for "> Python: Select Intepreter" and then press create enviroment (or using one you have already made).

Next run
```
pip install -r requirements.txt
```
In a terminal window to install the required packages.


## Basic model
To run our basic model, you will need to install the required packages to your device. You will also need the processed files from running the previous preprocessing script mentioned above. 

Then simply run the script 'logistic_regressor.py' and the training and results will be printed in the terminal.

Note that if you want to run the model test on the Liar test dataset, you will have to change the 'default_test_csv' variable at the bottom of the script to 'reduced_liar.csv'. 
And also 'y_test' variable to look for the 'type' column, instead of 'label', since the liar dataset uses a different name for labels. 

## Advanced model
You will need to install the required packages to your device. 

Firstly, run the script: 'Preprocess.py' which will yield the necessary data splits to run the model.

Then you need to run the script 'convert_to_lesser.py' to reduce the size of all the required files that we need to train and evaluate the model on. 

Then when you have completed these steps you can run the 'advanced_model.py' scripts and the results will be printed in the terminal.
