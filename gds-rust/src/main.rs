use csv::Reader;
use std::collections::HashSet;
// use std::fs::File;
use std::error::Error;
use regex::Regex;
use csv::Writer;
use once_cell::sync::Lazy;
use std::time::Instant;
use rust_stemmers::{Algorithm, Stemmer};
use indicatif::{ProgressBar, ProgressStyle};

static RE_DATE: Lazy<Regex> = Lazy::new(|| Regex::new(
    r"(?i)\b\d{4}-\d{2}-\d{2}\b|\b\d{2}/\d{2}/\d{4}\b|\b\d{2}\.\d{2}\.\d{4}\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s\d{1,2},?\s\d{4}\b|\b\d{1,2}\s(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|\b\d{1,2}(st|nd|rd|th)?\s(of\s)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b"
).unwrap());
static RE_EMAIL: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b").unwrap());
static RE_URL: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b(https?://[^\s]+)|(\bwww\.[^\s]+)").unwrap());
static RE_NUM: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b\d+\b").unwrap());
static RE_PUNCT: Lazy<Regex> = Lazy::new(|| Regex::new(r"[^a-zA-Z\s/]").unwrap());
static RE_WHITESPACE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+").unwrap());

// Function to tokenize the content
fn tokenize(content: &str) -> Vec<&str> {
    content.split_whitespace().collect()
}

fn load_stopwords(file_path: &str) -> HashSet<String> {
    std::fs::read_to_string(file_path)
        .unwrap_or_default()
        .lines()
        .map(|word| word.to_string()) // Convert `&str` to `String`
        .collect()
}

// Function to remove stopwords
fn remove_stopwords<'a>(tokens: Vec<&'a str>, stopwords: &HashSet<String>) -> Vec<&'a str> {
    tokens.into_iter()
        .filter(|word| !stopwords.contains(*word))
        .collect()
}

// Function to clean text
fn clean_text(text: &str) -> String {
    let mut cleaned_text = text.to_lowercase();

    // Replace patterns with explicit placeholders
    cleaned_text = RE_DATE.replace_all(&cleaned_text, " <DATE> ").to_string();
    cleaned_text = RE_EMAIL.replace_all(&cleaned_text, " <EMAIL> ").to_string();
    cleaned_text = RE_URL.replace_all(&cleaned_text, " <URL> ").to_string();
    cleaned_text = RE_NUM.replace_all(&cleaned_text, " <NUMBER> ").to_string();

    // Replace certain punctuation (e.g., '-') with spaces to avoid merging words
    cleaned_text = cleaned_text.replace('-', " ");

    // Split text into tokens, remove punctuation for non-placeholder words, and join back
    cleaned_text = cleaned_text
        .split_whitespace()
        .map(|word| {
            if word.starts_with('<') && word.ends_with('>') {
                word.to_string() // Preserve placeholders as-is
            } else {
                // Remove surrounding punctuation for other words
                RE_PUNCT.replace_all(word, "").to_string()
            }
        })
        .collect::<Vec<String>>()
        .join(" ");

    // Clean up extra whitespace
    cleaned_text = RE_WHITESPACE.replace_all(&cleaned_text, " ").to_string();

    cleaned_text.trim().to_string()
}

// Function to process and save with cleaned text
fn process_and_save(
    input_file: &str,
    output_file: &str,
    column_name: &str,
) -> Result<(), Box<dyn Error>> {
    let mut rdr = Reader::from_path(input_file)?;
    let mut wtr = Writer::from_path(output_file)?;

    // Write headers for the output file (original headers + new column)
    let mut headers = rdr.headers()?.clone();
    headers.push_field("content-tokens_stemmed");
    wtr.write_record(&headers)?;

    // Find the index of the column name
    let column_index = headers
        .iter()
        .position(|h| h == column_name)
        .ok_or_else(|| format!("Column '{}' not found in headers", column_name))?;

    // Prepare stopwords
    let stopwords: HashSet<String> = load_stopwords("stopwords.txt");

    let en_stemmer = Stemmer::create(Algorithm::English);

    // Create a progress bar
    let total_records = rdr.records().count(); // Count the total number of records
    let pb = ProgressBar::new(total_records as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({percent}%)")
            .unwrap()
            .progress_chars("##-"),
    );

    // Reset the reader since we iterated it to count records
    let mut rdr = csv::Reader::from_path(input_file)?;

    for result in rdr.records() {
        let record = result?;
        let content = record.get(column_index).unwrap_or(""); // Use the dynamic column index

        // Clean the text
        let cleaned_content = clean_text(content);

        // Tokenize and process the content
        let tokens = tokenize(&cleaned_content);
        let filtered_tokens = remove_stopwords(tokens, &stopwords);
        let stemmed_tokens = filtered_tokens
            .into_iter()
            .map(|word| en_stemmer.stem(word).to_string())
            .collect::<Vec<String>>();
        let processed_content = stemmed_tokens.join(" "); // Combine filtered tokens back into a single string

        // Build the output record (original + processed content)
        let mut new_record = record.clone();
        new_record.push_field(&processed_content);

        // Write the new record to the output file
        wtr.write_record(&new_record)?;

        // Update the progress bar
        pb.inc(1);
    }

    pb.finish_with_message("Processing complete."); // Complete the progress bar
    wtr.flush()?; // Ensure all data is written to the file

    println!(
        "Processed data saved with cleaned and concatenated content to {}",
        output_file
    );

    Ok(())
}

fn main() {
    let input_file = "../data/995,000_rows.csv"; // Replace with your input CSV file path
    let output_file = "../data/995,000_rows_processed.csv"; // Replace with your desired output file name
    let column_name = "content"; // Replace with the name of the column you want to process

    // Start timing
    let start = Instant::now();

    if let Err(err) = process_and_save(input_file, output_file, column_name) {
        eprintln!("Error: {}", err);
    }

    let duration = start.elapsed();
    println!("Time taken to process the file: {:.2?}", duration);
}
