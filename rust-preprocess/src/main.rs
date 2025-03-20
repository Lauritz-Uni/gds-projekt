// Import required crates and modules
use csv::{Reader, Writer};
use fxhash::FxHashSet;  // Faster hash implementation
use indicatif::{ProgressBar, ProgressStyle};  // Progress bar utilities
use once_cell::sync::Lazy;  // For lazy initialization of static variables
use rayon::prelude::*;  // Parallel processing utilities
use regex::Regex;
use rust_stemmers::{Algorithm, Stemmer};  // Stemming functionality
use std::error::Error;
use std::time::Instant;
use clap::Parser;  // Command-line argument parsing
use clap::ArgAction;
// Additional imports for random shuffling and path manipulation
use rand::seq::SliceRandom;
use rand::rng;
use std::path::Path;
use std::sync::Arc;

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

    /// Keep processed data in the output CSV (if false, will not create example_processed.csv)
    #[arg(short, long, action=ArgAction::SetTrue)]
    keep_processed: bool,
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

// Regex for punctuation removal (keeping letters, spaces, and angle brackets)
static RE_UNUSED: Lazy<Regex> = Lazy::new(|| Regex::new(r"[^a-zA-Z <>]").unwrap());
// Regex for normalizing whitespace
static RE_WHITESPACE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+").unwrap());

const STOPWORDS: &str = include_str!("../stopwords.txt");
/// Load stopwords from a file into a hash set
fn load_stopwords() -> Result<FxHashSet<String>, Box<dyn Error>> {
    Ok(STOPWORDS
        .lines()
        .map(|s| s.trim().to_lowercase())
        .collect())
}

/// Split text into individual tokens (words)
fn tokenize(content: &str) -> Vec<String> {
    content.split_ascii_whitespace()
        .map(|s: &str| s.to_string())
        .collect()
}

/// Remove stopwords from a list of tokens
fn remove_stopwords(tokens: Vec<String>, stopwords: &FxHashSet<String>) -> Vec<String> {
    tokens.into_iter()
        .filter(|word| {
            // Skip placeholder tokens and check lowercase for others
            if word.starts_with('<') && word.ends_with('>') {
                true
            } else {
                !stopwords.contains(word)
            }
        })
        .collect()
}

fn format_numbers_preserving_format(s: &str) -> String {
    s.split(';')
        .map(|part| {
            part.trim()
                .parse::<f64>()
                .map(|n| {
                    if n.fract() == 0.0 {
                        format!("{:.0}", n) // Integer format
                    } else {
                        n.to_string() // Keep original float
                    }
                })
                .unwrap_or_else(|_| part.to_string()) // Non-numeric values
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// Clean and process text with multiple steps:
/// 1. Entity extraction (dates, emails, URLs, numbers)
/// 2. Punctuation removal
/// 3. Whitespace normalization
/// Returns cleaned text and extracted entities
fn clean_text(text: &str) -> (String, Vec<String>, Vec<String>, Vec<String>, Vec<String>) {
    let text = text.to_lowercase();
    
    let mut dates = Vec::new();
    let mut emails = Vec::new();
    let mut urls = Vec::new();
    let mut numbers = Vec::new();

    let cleaned_text = RE_COMBINED.replace_all(&text, |caps: &regex::Captures<'_>| {
        if let Some(m) = caps.get(1) {
            dates.push(m.as_str().to_string());
            " <DATE> "
        } else if let Some(m) = caps.get(2) {
            emails.push(m.as_str().to_string());
            " <EMAIL> "
        } else if let Some(m) = caps.get(3) {
            urls.push(m.as_str().to_string());
            " <URL> "
        } else if let Some(m) = caps.get(4) {
            numbers.push(m.as_str().to_string());
            " <NUMBER> "
        } else {
            ""
        }
    }).to_string();

    let cleaned = RE_WHITESPACE
        .replace_all(
            &RE_UNUSED.replace_all(&cleaned_text, " "),
            " "
        )
        .trim()
        .to_string();

    (cleaned, dates, emails, urls, numbers)
}

/// Main processing function that handles:
/// - Reading input CSV
/// - Parallel processing of records
/// - Writing output CSV with additional fields
fn process_and_save(
    input_file: &str,
    output_file: &str,
    column_name: &str,
    keep_processed: &bool
) -> Result<(), Box<dyn Error>> {
    let processing_start = Instant::now();


    println!("Reading {input_file}!");
    // Read input CSV file
    let mut rdr = Reader::from_path(input_file)?;
    let string_records: Vec<csv::StringRecord> = rdr.records().collect::<Result<_, _>>()?;
    let total_records = string_records.len();
    println!("Time taken to read data: {:.2?}", processing_start.elapsed());

    let process_records_start = Instant::now();
    // Prepare output CSV writer with extended headers
    let mut headers = rdr.headers()?.clone();
    headers.push_field("dates");
    headers.push_field("emails");
    headers.push_field("urls");
    headers.push_field("numbers");
    headers.push_field(&(column_name.to_owned()+"-tokens"));
    headers.push_field(&(column_name.to_owned()+"-tokens_no_stop"));
    headers.push_field(&(column_name.to_owned()+"-tokens_stemmed"));

    // Find index of the target column
    let column_index = headers
        .iter()
        .position(|h| h == column_name)
        .ok_or_else(|| format!("Column '{}' not found", column_name))?;

    // Load resources needed for processing
    let stopwords = load_stopwords()?;
    let en_stemmer = Stemmer::create(Algorithm::English);
    
    // Setup progress bar with thread-safe wrapper
    let pb = Arc::new(ProgressBar::new(total_records as u64));
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40} {pos}/{len} ({percent}%) ETA: {eta_precise}")?
            .progress_chars("##-"),
    );

    // Process records in parallel using Rayon
    let processed_records: Vec<_> = string_records
    .par_iter()
        .filter_map(|record| {
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

            // Update progress bar for each processed record
            pb.inc(1);

            // Skip records with empty stemmed
            if stemmed.is_empty() {
                return None;
            }

            // Build enhanced output record
            let mut new_record = record.clone();
            new_record.push_field(&format_numbers_preserving_format(
                &dates.join("; ")
            ));
            new_record.push_field(&format_numbers_preserving_format(
                &emails.join("; ")
            ));
            new_record.push_field(&format_numbers_preserving_format(
                &urls.join("; ")
            ));
            new_record.push_field(&format_numbers_preserving_format(
                &numbers.join("; ")
            ));
            new_record.push_field(&tokens.join(" "));
            new_record.push_field(&filtered.join(" "));
            new_record.push_field(&stemmed);
            
            Some(new_record)
        })
        .collect();
    
    drop(rdr);
    drop(string_records);

    pb.finish_with_message("Done processing");
    println!("Time taken to process data: {:.2?}", process_records_start.elapsed());

    // Write processed records to output CSV, if keep_processed is true
    if *keep_processed {
        let write_start = Instant::now();
        if let Err(e) = std::fs::remove_file(output_file) {
            eprintln!("Error removing file: {}", e);
        }
        let mut wtr = Writer::from_path(output_file)?;
        wtr.write_record(&headers)?;
        for record in &processed_records {
            wtr.write_record(&*record)?;
        }
        wtr.flush()?;
        drop(wtr);
        println!("Time taken to write data: {:.2?}", write_start.elapsed());
    }

    // Shuffle records for random split
    let mut processed_records = processed_records;
    let mut rng = rng();
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
        .unwrap_or("./output");
    let extension = output_path.extension()
        .map(|e| e.to_str().unwrap_or("csv"))
        .unwrap_or("csv");

    let train_path = output_path.with_file_name(format!("{}_train.{}", stem, extension));
    let val_path = output_path.with_file_name(format!("{}_val.{}", stem, extension));
    let test_path = output_path.with_file_name(format!("{}_test.{}", stem, extension));

    // Determine columns to exclude
    let exclude_names = vec![
        column_name.to_string(),
        format!("{}-tokens", column_name),
        format!("{}-tokens_no_stop", column_name),
    ];

    let exclude_indices: Vec<usize> = headers.iter()
        .enumerate()
        .filter(|(_, name)| exclude_names.contains(&name.to_string()))
        .map(|(i, _)| i)
        .collect();

    let reliability_column = "type";
    let reliability_col_index = headers.iter()
        .position(|h| h == reliability_column)
        .ok_or_else(|| format!("Column '{}' not found", reliability_column))?;
    
    // Modified: Create filtered headers with type column
    let mut filtered_header_fields: Vec<&str> = headers.iter()
        .enumerate()
        .filter(|(i, _)| !exclude_indices.contains(i))
        .map(|(_, name)| name)
        .collect();
        filtered_header_fields.push("label"); // Add type column
    let filtered_headers = csv::StringRecord::from(filtered_header_fields);

    // Add a new progress bar for writing
    let write_pb = ProgressBar::new(3);  // 3 splits
    write_pb.set_style(ProgressStyle::default_bar()
        .template("[Writing splits] {bar:40} {pos}/{len} ({eta_precise})")?);

    // Write training split
    {
        let mut train_wtr = Writer::from_path(train_path)?;
        train_wtr.write_record(&filtered_headers)?;
        for record in &processed_records[0..train_size] {
            // Get type from original record
            let reliability_status = record.get(reliability_col_index).unwrap_or_default();
            let type_label = if reliability_status.eq_ignore_ascii_case("reliable") {
                "reliable"
            } else {
                "fake"
            };

            // Build filtered record
            let mut filtered_record: Vec<&str> = record.iter()
                .enumerate()
                .filter(|(i, _)| !exclude_indices.contains(i))
                .map(|(_, field)| field)
                .collect();
            
            // Append type column
            filtered_record.push(type_label);
            
            train_wtr.write_record(&filtered_record)?;
        }
        train_wtr.flush()?;
        drop(train_wtr);
    }

    write_pb.inc(1);

    // Write validation split
    {
        if let Err(_e) = std::fs::remove_file(&val_path) {
            // println!("Failed to remove file: {}", e);
        }
        let mut val_wtr = Writer::from_path(val_path)?;
        val_wtr.write_record(&filtered_headers)?;
        for record in &processed_records[train_size..train_size + val_size] {
            // Get type status from original record
            let reliability_status = record.get(reliability_col_index).unwrap_or_default();
            let type_label = if reliability_status.eq_ignore_ascii_case("reliable") {
                "reliable"
            } else {
                "fake"
            };

            let mut filtered_record: Vec<&str> = record.iter()
                .enumerate()
                .filter(|(i, _)| !exclude_indices.contains(i))
                .map(|(_, field)| field)
                .collect();
            
            // Append type column
            filtered_record.push(type_label);

            val_wtr.write_record(&filtered_record)?;
        }
        val_wtr.flush()?;
        drop(val_wtr);
    }

    write_pb.inc(1);

    // Write test split
    {
        if let Err(_e) = std::fs::remove_file(&test_path) {
            // println!("Failed to remove file: {}", e);
        }
        let mut test_wtr = Writer::from_path(test_path)?;
        test_wtr.write_record(&filtered_headers)?;
        for record in &processed_records[train_size + val_size..] {
            // Get type status from original record
            let reliability_status = record.get(reliability_col_index).unwrap_or_default();
            let type_label = if reliability_status.eq_ignore_ascii_case("reliable") {
                "reliable"
            } else {
                "fake"
            };

            let mut filtered_record: Vec<&str> = record.iter()
                .enumerate()
                .filter(|(i, _)| !exclude_indices.contains(i))
                .map(|(_, field)| field)
                .collect();
            
            // Append type column
            filtered_record.push(type_label);

            test_wtr.write_record(&filtered_record)?;
        }
        test_wtr.flush()?;
        drop(test_wtr);
    }

    write_pb.inc(1);

    write_pb.finish_with_message("Finished writing splits");

    drop(stopwords);

    println!("Files successfully closed");
    
    println!("Total processing time: {:.2?}", processing_start.elapsed());
    Ok(())
}

/// Main function handling command-line execution
fn main() {
    let args = Args::parse();
    
    let start = Instant::now();
    println!("Processing Started!");

    if args.keep_processed {
        println!("Keeping processed files...");
    }

    if let Err(e) = process_and_save(&args.input, &args.output, &args.column, &args.keep_processed) {
        eprintln!("Error: {}", e);
    }
    println!("Processing Done! Total time: {:.2?}", start.elapsed());
}