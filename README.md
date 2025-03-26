
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
To open the project in VSCode

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

# Models
Our models are written in Python 3.12.x. It is assumed that you have any Python 3.12.x installed and that you are using it to run the files.

## Virtual enviroment initialization
You will probably want to create a virtual enviroment. We used Conda while coding this project, but most other types of Virtual enviroments should work.

If you are using VSCode, you can do this by opening the command pallet and searching for "> Python: Select Intepreter" and then press create enviroment (or using one you have already made).

Next run
```
pip install -r requirements.txt
```
In a terminal window to install the required packages.


## Basic model

## Advanced model
