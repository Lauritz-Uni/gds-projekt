// Import required crates and modules
use csv::{Reader, Writer};
use fxhash::FxHashSet;  // Faster hash implementation
use indicatif::{ProgressBar, ProgressStyle};  // Progress bar utilities
use once_cell::sync::Lazy;  // For lazy initialization of static variables
use rayon::prelude::*;  // Parallel processing utilities
use regex::Regex;
use rust_stemmers::{Algorithm, Stemmer};  // Stemming functionality
use std::error::Error;
use std::sync::Mutex;  // For thread-safe progress bar updates
use std::time::Instant;
use clap::Parser;  // Command-line argument parsing
// Additional imports for random shuffling and path manipulation
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::path::{Path, PathBuf};

// Define command-line arguments structure using clap
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

// Compile regular expressions once using lazy initialization
static RE_COMBINED: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?xi)
        # Combined pattern for various entity types:
        # Dates in multiple formats
        (\b\d{4}-\d{2}-\d{2}\b|
        \b\d{2}/\d{2}/\d{4}\b|
        \b\d{2}\.\d{2}\.\d{4}\b|
        \b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s\d{1,2},?\s\d{4}\b|
        \b\d{1,2}\s(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|
        \b\d{1,2}(?:st|nd|rd|th)?\s(?:of\s)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b)|
        
        # Email addresses
        (\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b)|
        
        # URL patterns (including domains and paths)
        (
            https?://[^\s)\]}'<>"]+|    # HTTP URLs
            www\.[^\s)\]}'<>"]+|        # www URLs
            \b[a-z0-9-]+\.[a-z]{2,}(?:/[^\s)\]}'<>"]*)*  # Domain paths
        )|
        
        # Numeric values
        (\b\d+\b)
    "#).expect("Failed to compile regex")
});

// Regex for punctuation removal (keeping letters, spaces, and slashes)
static RE_PUNCT: Lazy<Regex> = Lazy::new(|| Regex::new(r"[^a-zA-Z\s/]").unwrap());
// Regex for normalizing whitespace
static RE_WHITESPACE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+").unwrap());

/// Load stopwords from a file into a hash set
fn load_stopwords(file_path: &str) -> Result<FxHashSet<String>, Box<dyn Error>> {
    Ok(std::fs::read_to_string(file_path)?
        .lines()
        .map(|s: &str| s.to_lowercase())
        .collect())
}

/// Split text into individual tokens (words)
fn tokenize(content: &str) -> Vec<String> {
    content.split_whitespace()
        .map(|s: &str| s.to_string())
        .collect()
}

/// Remove stopwords from a list of tokens
fn remove_stopwords(tokens: Vec<String>, stopwords: &FxHashSet<String>) -> Vec<String> {
    tokens.into_iter()
        .filter(|word| !stopwords.contains(&word.to_lowercase()))
        .collect()
}

/// Clean and process text with multiple steps:
/// 1. Entity extraction (dates, emails, URLs, numbers)
/// 2. Punctuation removal
/// 3. Whitespace normalization
/// Returns cleaned text and extracted entities
fn clean_text(text: &str) -> (String, Vec<String>, Vec<String>, Vec<String>, Vec<String>) {
    let text = text.to_lowercase();
    
    // Vectors to store extracted entities
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

    // Second pass: Clean punctuation and normalize whitespace
    let cleaned_text: String = RE_WHITESPACE
        .replace_all(
            &cleaned_text
                .split_whitespace()
                .map(|word: &str| {
                    // Preserve placeholder tokens
                    if word.starts_with('<') && word.ends_with('>') {
                        word.to_string()
                    } else {
                        RE_PUNCT.replace_all(word, "").into_owned()
                    }
                })
                .filter(|s: &String| !s.is_empty())
                .collect::<Vec<_>>()
                .join(" "),
            " ",
        )
        .trim()
        .to_string();

    (cleaned_text, dates, emails, urls, numbers)
}

/// Main processing function that handles:
/// - Reading input CSV
/// - Parallel processing of records
/// - Writing output CSV with additional fields
fn process_and_save(
    input_file: &str,
    output_file: &str,
    column_name: &str,
) -> Result<(), Box<dyn Error>> {
    let processing_start = Instant::now();

    // Read input CSV file
    let mut rdr = Reader::from_path(input_file)?;
    let records: Vec<_> = rdr.records().collect::<Result<_, _>>()?;
    let total_records = records.len();
    println!("Time taken to read data: {:.2?}", processing_start.elapsed());

    // Prepare output CSV writer with extended headers
    let mut wtr = Writer::from_path(output_file)?;
    let mut headers = rdr.headers()?.clone();
    headers.push_field("dates");
    headers.push_field("emails");
    headers.push_field("urls");
    headers.push_field("numbers");
    headers.push_field(&(column_name.to_owned()+"-tokens"));
    headers.push_field(&(column_name.to_owned()+"-tokens_no_stop"));
    headers.push_field(&(column_name.to_owned()+"-tokens_stemmed"));
    wtr.write_record(&headers)?;

    // Find index of the target column
    let column_index = headers
        .iter()
        .position(|h| h == column_name)
        .ok_or_else(|| format!("Column '{}' not found", column_name))?;

    // Load resources needed for processing
    let stopwords = load_stopwords("stopwords.txt")?;
    let en_stemmer = Stemmer::create(Algorithm::English);
    
    // Setup progress bar with thread-safe wrapper
    let pb = ProgressBar::new(total_records as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40} {pos}/{len} ({percent}%) ETA: {eta_precise}")?
            .progress_chars("##-"),
    );
    let pb_mutex = Mutex::new(pb);

    // Process records in parallel using Rayon
    let processed_records: Vec<_> = records
        .par_iter()
        .map(|record| {
            let content = record.get(column_index).unwrap_or("");
            
            // Clean text and extract entities
            let (cleaned, dates, emails, urls, numbers) = clean_text(content);
            
            // Tokenization and text processing
            let tokens = tokenize(&cleaned);
            let filtered = remove_stopwords(tokens.clone(), &stopwords);
            let stemmed = filtered.iter()
                .map(|word| en_stemmer.stem(word).to_string())
                .collect::<Vec<_>>()
                .join(" ");

            // Build enhanced output record
            let mut new_record = record.clone();
            new_record.push_field(&dates.join("; "));
            new_record.push_field(&emails.join("; "));
            new_record.push_field(&urls.join("; "));
            new_record.push_field(&numbers.join("; "));
            new_record.push_field(&tokens.join(" "));
            new_record.push_field(&filtered.join(" "));
            new_record.push_field(&stemmed);
            
            // Update progress bar
            pb_mutex.lock().unwrap().inc(1);
            
            new_record
        })
        .collect();

    // Shuffle records for random split
    let mut processed_records = processed_records;
    let mut rng = thread_rng();
    processed_records.as_mut_slice().shuffle(&mut rng);

    // Calculate split sizes
    let total = processed_records.len();
    let train_size = (total as f64 * 0.8) as usize;
    let val_size = (total as f64 * 0.1) as usize;
    // let test_size = total - train_size - val_size;

    // Create split paths
    let output_path = Path::new(output_file);
    let stem = output_path.file_stem()
        .unwrap_or_default()
        .to_str()
        .unwrap_or("output");
    let extension = output_path.extension()
        .map(|e| e.to_str().unwrap_or("csv"))
        .unwrap_or("csv");

        let train_path = output_path.with_file_name(format!("{}_train.{}", stem, extension));
        let val_path = output_path.with_file_name(format!("{}_val.{}", stem, extension));
        let test_path = output_path.with_file_name(format!("{}_test.{}", stem, extension));
    
        // Write training split
        let mut train_wtr = Writer::from_path(train_path)?;
        train_wtr.write_record(&headers)?;
        for record in &processed_records[0..train_size] {
            train_wtr.write_record(record)?;
        }
        train_wtr.flush()?;
    
        // Write validation split
        let mut val_wtr = Writer::from_path(val_path)?;
        val_wtr.write_record(&headers)?;
        for record in &processed_records[train_size..train_size + val_size] {
            val_wtr.write_record(record)?;
        }
        val_wtr.flush()?;
    
        // Write test split
        let mut test_wtr = Writer::from_path(test_path)?;
        test_wtr.write_record(&headers)?;
        for record in &processed_records[train_size + val_size..] {
            test_wtr.write_record(record)?;
        }
        test_wtr.flush()?;
    
    pb_mutex.lock().unwrap().finish_with_message("Done with file: {output_file}");
    wtr.flush()?;
    println!("Total processing time: {:.2?}", processing_start.elapsed());
    Ok(())
}

/// Main function handling command-line execution
fn main() {
    let args = Args::parse();
    
    let start = Instant::now();
    if let Err(e) = process_and_save(&args.input, &args.output, &args.column) {
        eprintln!("Error: {}", e);
    }
    println!("Total time: {:.2?}", start.elapsed());
}