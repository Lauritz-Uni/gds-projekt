use csv::{Reader, Writer};
use fxhash::FxHashSet;
use indicatif::{ProgressBar, ProgressStyle};
use once_cell::sync::Lazy;
use rayon::prelude::*;
use regex::Regex;
use rust_stemmers::{Algorithm, Stemmer};
use std::error::Error;
use std::sync::Mutex;
use std::time::Instant;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input CSV file path
    #[arg(short, long)]
    input: String,

    /// Output CSV file path
    #[arg(short, long)]
    output: String,

    /// Name of the column to process
    #[arg(short, long, default_value = "content")]
    column: String,
}

static RE_COMBINED: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?xi)
        # Dates
        (\b\d{4}-\d{2}-\d{2}\b|
        \b\d{2}/\d{2}/\d{4}\b|
        \b\d{2}\.\d{2}\.\d{4}\b|
        \b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s\d{1,2},?\s\d{4}\b|
        \b\d{1,2}\s(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|
        \b\d{1,2}(?:st|nd|rd|th)?\s(?:of\s)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b)|
        
        # Emails
        (\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b)|
        
        # URLs (fixed)
        (
            https?://[^\s)\]}'<>"]+|    # HTTP URLs
            www\.[^\s)\]}'<>"]+|        # www URLs
            \b[a-z0-9-]+\.[a-z]{2,}(?:/[^\s)\]}'<>"]*)*  # Domain paths
        )|
        
        # Numbers
        (\b\d+\b)
    "#).expect("Failed to compile regex")
});

static RE_PUNCT: Lazy<Regex> = Lazy::new(|| Regex::new(r"[^a-zA-Z\s/]").unwrap());
static RE_WHITESPACE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+").unwrap());

fn load_stopwords(file_path: &str) -> Result<FxHashSet<String>, Box<dyn Error>> {
    Ok(std::fs::read_to_string(file_path)?
        .lines()
        .map(|s| s.to_lowercase())
        .collect())
}


fn tokenize(content: &str) -> Vec<String> {
    content.split_whitespace()
        .map(|s| s.to_string())
        .collect()
}


fn remove_stopwords(tokens: Vec<String>, stopwords: &FxHashSet<String>) -> Vec<String> {
    tokens.into_iter()
        .filter(|word| !stopwords.contains(&word.to_lowercase()))
        .collect()
}


fn clean_text(text: &str) -> (String, Vec<String>, Vec<String>, Vec<String>, Vec<String>) {
    let text = text.to_lowercase();
    
    let mut dates = Vec::new();
    let mut emails = Vec::new();
    let mut urls = Vec::new();
    let mut numbers = Vec::new();

    // First pass: Extract entities and replace with placeholders
    let cleaned_text = RE_COMBINED.replace_all(&text, |caps: &regex::Captures<'_>| {
        if let Some(m) = caps.get(1) { // Date
            dates.push(m.as_str().to_string());
            " <DATE> "
        } else if let Some(m) = caps.get(2) { // Email
            emails.push(m.as_str().to_string());
            " <EMAIL> "
        } else if let Some(m) = caps.get(3) { // URL
            urls.push(m.as_str().to_string());
            " <URL> "
        } else if let Some(m) = caps.get(4) { // Number
            numbers.push(m.as_str().to_string());
            " <NUMBER> "
        } else {
            ""
        }
    }).to_string();

    let cleaned_text = RE_WHITESPACE
        .replace_all(
            &cleaned_text
                .split_whitespace()
                .map(|word| {
                    if word.starts_with('<') && word.ends_with('>') {
                        word.to_string()
                    } else {
                        RE_PUNCT.replace_all(word, "").into_owned()
                    }
                })
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join(" "),
            " ",
        )
        .trim()
        .split_whitespace()
        .map(ToString::to_string)
        .collect();
    
    (cleaned_text, dates, emails, urls, numbers)
}

fn process_and_save(
    input_file: &str,
    output_file: &str,
    column_name: &str,
) -> Result<(), Box<dyn Error>> {
    let start = Instant::now();
    let mut rdr = Reader::from_path(input_file)?;
    let records: Vec<_> = rdr.records().collect::<Result<_, _>>()?;
    let total_records = records.len();
    println!("Time taken to read data: {:.2?}", start.elapsed());

    let mut wtr = Writer::from_path(output_file)?;
    let mut headers = rdr.headers()?.clone();
    headers.push_field("content-tokens_stemmed");
    headers.push_field("dates");
    headers.push_field("emails");
    headers.push_field("urls");
    headers.push_field("numbers");
    wtr.write_record(&headers)?;

    let column_index = headers
        .iter()
        .position(|h| h == column_name)
        .ok_or_else(|| format!("Column '{}' not found", column_name))?;

    let stopwords = load_stopwords("stopwords.txt")?;
    let en_stemmer = Stemmer::create(Algorithm::English);
    let pb = ProgressBar::new(total_records as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40} {pos}/{len} ({percent}%) ETA: {eta_precise}")?
            .progress_chars("##-"),
    );
    let pb_mutex = Mutex::new(pb);

    let processed_records: Vec<_> = records
        .par_iter()
        .map(|record| {
            let content = record.get(column_index).unwrap_or("");
            
            // Get cleaned text and entities
            let (cleaned, dates, emails, urls, numbers) = clean_text(content);
            
            // Existing processing
            let tokens = tokenize(&cleaned);
            let filtered = remove_stopwords(tokens, &stopwords);
            let stemmed = filtered.iter()
                .map(|word| en_stemmer.stem(word).to_string())
                .collect::<Vec<_>>()
                .join(" ");

            // Build new record with additional fields
            let mut new_record = record.clone();
            new_record.push_field(&stemmed);
            new_record.push_field(&dates.join("; "));
            new_record.push_field(&emails.join("; "));
            new_record.push_field(&urls.join("; "));
            new_record.push_field(&numbers.join("; "));
            
            new_record
        })
        .collect();

    for record in processed_records {
        wtr.write_record(&record)?;
    }
    
    pb_mutex.lock().unwrap().finish_with_message("Done with file: {output_file}");
    wtr.flush()?;
    println!("Total processing time: {:.2?}", start.elapsed());
    Ok(())
}

fn main() {
    let args = Args::parse();
    
    let start = Instant::now();
    if let Err(e) = process_and_save(&args.input, &args.output, &args.column) {
        eprintln!("Error: {}", e);
    }
    println!("Total time: {:.2?}", start.elapsed());
}