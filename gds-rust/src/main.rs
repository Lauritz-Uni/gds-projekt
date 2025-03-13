use csv::Reader;
use std::collections::HashSet;
use std::fs::File;
use std::error::Error;
use regex::Regex;
use csv::Writer;
use once_cell::sync::Lazy;

// Regex
static RE_DATE: Lazy<Regex> = Lazy::new(|| Regex::new(r"your_date_regex_here").unwrap());
static RE_EMAIL: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b").unwrap());
static RE_URL: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b(https?://[^\s]+)|(\bwww\.[^\s]+)").unwrap());
static RE_NUM: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b\d+\b").unwrap());
static RE_PUNCT: Lazy<Regex> = Lazy::new(|| Regex::new(r"[[:punct:]]").unwrap());
static RE_WHITESPACE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+").unwrap());

// Function to tokenize the content
fn tokenize(content: &str) -> Vec<&str> {
    content.split_whitespace().collect()
}

// Function to remove stopwords
fn remove_stopwords<'a>(tokens: Vec<&'a str>, stopwords: &'a HashSet<&'a str>) -> Vec<&'a str> {
    tokens.into_iter()
        .filter(|word| !stopwords.contains(*word))
        .collect()
}


fn clean_text(text: &str) -> String {
    let mut cleaned_text = text.to_lowercase();

    // Replace patterns
    cleaned_text = RE_DATE.replace_all(&cleaned_text, "<DATE>").to_string();
    cleaned_text = RE_EMAIL.replace_all(&cleaned_text, "<EMAIL>").to_string();
    cleaned_text = RE_URL.replace_all(&cleaned_text, "<URL>").to_string();
    cleaned_text = RE_NUM.replace_all(&cleaned_text, "<NUM>").to_string();

    // Remove punctuation without affecting placeholders
    cleaned_text = cleaned_text
        .split_whitespace()
        .map(|word| {
            if word.starts_with('<') && word.ends_with('>') {
                word.to_string()
            } else {
                RE_PUNCT.replace_all(word, "").to_string()
            }
        })
        .collect::<Vec<String>>()
        .join(" ");

    // Clean up whitespace
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
    headers.push_field("processed_content");
    wtr.write_record(&headers)?;

    // Find the index of the column name
    let column_index = headers
        .iter()
        .position(|h| h == column_name)
        .ok_or_else(|| format!("Column '{}' not found in headers", column_name))?;

    // Prepare stopwords
    let stopwords: HashSet<&str> = [
        "a", "an", "the", "is", "and", "or", "to", "of", "it", "this", "that", // Add more stopwords here
    ]
    .iter()
    .cloned()
    .collect();

    for result in rdr.records() {
        let record = result?;
        let content = record.get(column_index).unwrap_or(""); // Use the dynamic column index

        // Clean the text
        let cleaned_content = clean_text(content);

        // Tokenize and process the content
        let tokens = tokenize(&cleaned_content);
        let filtered_tokens = remove_stopwords(tokens, &stopwords);
        let processed_content = filtered_tokens.join(" "); // Combine filtered tokens back into a single string

        // Build the output record (original + processed content)
        let mut new_record = record.clone();
        new_record.push_field(&processed_content);

        // Write the new record to the output file
        wtr.write_record(&new_record)?;
    }

    wtr.flush()?; // Ensure all data is written to the file
    println!(
        "Processed data saved with cleaned and concatenated content to {}",
        output_file
    );

    Ok(())
}

fn main() {
    let input_file = "../data/news_sample.csv"; // Replace with your input CSV file path
    let output_file = "../data/processed_news_with_column.csv"; // Replace with your desired output file name
    let column_name = "content"; // Replace with the name of the column you want to process

    if let Err(err) = process_and_save(input_file, output_file, column_name) {
        eprintln!("Error: {}", err);
    }
}
