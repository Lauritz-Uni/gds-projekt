# Grundlægende data science projekt

Run in a terminal window:
```
git clone https://github.com/Lauritz-Uni/gds-projekt.git
cd gds-projekt
code .
```

After that you want to create a virtual enviroment with conda. This can be done in vscode, by opening the command pallet and searching for "> Python: Select Intepreter" and then press create enviroment (or use one you have already made).

Next run
```
pip install -r requirements.txt
```
In a terminal window to install the required packages.

## Preprocessor
To compile preprocessor, run this in a terminal window in rust-preprocess folder
```
cargo build --release
```
This will take some time to compile

To preprocess a file run this in a terminal window
```
./rust-preprocess/target/release/rust-preprocess.exe --input "./data/995,000_rows.csv" --output "./output/995,000_rows_processed.csv"
```
Note: this creates alot of additional data, so make sure you have enough space


## Task 1

Tokenize text
1) lowercase
2) remove duplicate whitespace characters
3) numbers, dates, emails, and URLs should be replaced by "<NUM>", "<DATE>", "<EMAIL>" AND "<URL>"
4) 
