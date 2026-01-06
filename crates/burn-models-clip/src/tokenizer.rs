//! CLIP BPE Tokenizer
//!
//! Implements the Byte Pair Encoding tokenizer used by CLIP models.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use regex::Regex;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid vocabulary file format")]
    InvalidVocab,

    #[error("Regex error: {0}")]
    Regex(#[from] regex::Error),
}

/// Special token IDs
pub const START_OF_TEXT: u32 = 49406;
pub const END_OF_TEXT: u32 = 49407;

/// Embedded CLIP BPE merges (from OpenAI's CLIP)
const CLIP_BPE_MERGES: &str = include_str!("../data/bpe_merges.txt");

/// CLIP BPE Tokenizer
pub struct ClipTokenizer {
    byte_encoder: HashMap<u8, char>,
    byte_decoder: HashMap<char, u8>,
    encoder: HashMap<String, u32>,
    decoder: HashMap<u32, String>,
    bpe_ranks: HashMap<(String, String), usize>,
    cache: std::cell::RefCell<HashMap<String, String>>,
    pat: Regex,
}

impl ClipTokenizer {
    /// Create a new tokenizer with the standard CLIP vocabulary (embedded)
    ///
    /// This uses the same BPE merges as OpenAI's CLIP model, which is standard
    /// for all Stable Diffusion 1.x and 2.x models.
    pub fn new() -> Self {
        Self::from_vocab(CLIP_BPE_MERGES)
            .expect("embedded CLIP vocabulary should be valid")
    }

    /// Create a new tokenizer from a vocabulary file
    pub fn from_file<P: AsRef<Path>>(vocab_path: P) -> Result<Self, TokenizerError> {
        let vocab_content = fs::read_to_string(vocab_path)?;
        Self::from_vocab(&vocab_content)
    }

    /// Create a new tokenizer from vocabulary string content
    pub fn from_vocab(vocab: &str) -> Result<Self, TokenizerError> {
        let byte_encoder = bytes_to_unicode();
        let byte_decoder: HashMap<char, u8> = byte_encoder.iter().map(|(&k, &v)| (v, k)).collect();

        // Parse BPE merges from vocabulary file
        let lines: Vec<&str> = vocab.lines().collect();

        // Skip header line if present (starts with #version)
        let merges_start = if lines.first().map_or(false, |l| l.starts_with("#version")) {
            1
        } else {
            0
        };

        let bpe_ranks: HashMap<(String, String), usize> = lines[merges_start..]
            .iter()
            .enumerate()
            .filter_map(|(i, line)| {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() == 2 {
                    Some(((parts[0].to_string(), parts[1].to_string()), i))
                } else {
                    None
                }
            })
            .collect();

        // Build encoder vocabulary
        let mut encoder = HashMap::new();
        let mut vocab_idx = 0u32;

        // Add byte-level tokens
        for c in byte_encoder.values() {
            encoder.insert(c.to_string(), vocab_idx);
            vocab_idx += 1;
        }

        // Add byte-level tokens with end-of-word marker
        for c in byte_encoder.values() {
            encoder.insert(format!("{}</w>", c), vocab_idx);
            vocab_idx += 1;
        }

        // Add merged tokens
        for (pair, _) in &bpe_ranks {
            encoder.insert(format!("{}{}", pair.0, pair.1), vocab_idx);
            vocab_idx += 1;
        }

        // Add special tokens
        encoder.insert("<|startoftext|>".to_string(), START_OF_TEXT);
        encoder.insert("<|endoftext|>".to_string(), END_OF_TEXT);

        let decoder: HashMap<u32, String> = encoder.iter().map(|(k, &v)| (v, k.clone())).collect();

        // Regex pattern for tokenization (matches CLIP's pattern)
        let pat = Regex::new(
            r"(?i)<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"
        )?;

        Ok(Self {
            byte_encoder,
            byte_decoder,
            encoder,
            decoder,
            bpe_ranks,
            cache: std::cell::RefCell::new(HashMap::new()),
            pat,
        })
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();

        // Normalize text
        let text = text
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase();

        // Find all matches
        for mat in self.pat.find_iter(&text) {
            let token = mat.as_str();

            // Convert to byte encoding
            let byte_encoded: String = token
                .bytes()
                .map(|b| self.byte_encoder.get(&b).copied().unwrap_or('?'))
                .collect();

            // Apply BPE
            let bpe_result = self.bpe(&byte_encoded);

            // Look up token IDs
            for bpe_token in bpe_result.split_whitespace() {
                if let Some(&id) = self.encoder.get(bpe_token) {
                    tokens.push(id);
                }
            }
        }

        tokens
    }

    /// Encode text with start/end tokens and padding to max_length
    pub fn encode_padded(&self, text: &str, max_length: usize) -> Vec<u32> {
        let mut tokens = vec![START_OF_TEXT];
        tokens.extend(self.encode(text));
        tokens.push(END_OF_TEXT);

        // Pad or truncate to max_length
        tokens.resize(max_length, 0);
        tokens
    }

    /// Decode token IDs back to text
    pub fn decode(&self, tokens: &[u32]) -> String {
        let text: String = tokens
            .iter()
            .filter_map(|&id| self.decoder.get(&id))
            .cloned()
            .collect();

        // Convert byte encoding back to text
        let bytes: Vec<u8> = text
            .chars()
            .filter_map(|c| self.byte_decoder.get(&c).copied())
            .collect();

        String::from_utf8_lossy(&bytes)
            .replace("</w>", " ")
            .trim()
            .to_string()
    }

    /// Apply BPE to a token
    fn bpe(&self, token: &str) -> String {
        // Check cache
        if let Some(cached) = self.cache.borrow().get(token) {
            return cached.clone();
        }

        let mut word: Vec<String> = token.chars().map(|c| c.to_string()).collect();

        if word.is_empty() {
            return String::new();
        }

        // Add end-of-word marker to last character
        if let Some(last) = word.last_mut() {
            *last = format!("{}</w>", last);
        }

        // Iteratively merge pairs
        loop {
            // Find the pair with the lowest rank
            let pairs = get_pairs(&word);
            if pairs.is_empty() {
                break;
            }

            let min_pair = pairs
                .iter()
                .filter_map(|pair| self.bpe_ranks.get(pair).map(|&rank| (pair, rank)))
                .min_by_key(|&(_, rank)| rank);

            let Some((bigram, _)) = min_pair else {
                break;
            };

            // Merge the pair
            let mut new_word = Vec::new();
            let mut i = 0;

            while i < word.len() {
                if i < word.len() - 1 && word[i] == bigram.0 && word[i + 1] == bigram.1 {
                    new_word.push(format!("{}{}", bigram.0, bigram.1));
                    i += 2;
                } else {
                    new_word.push(word[i].clone());
                    i += 1;
                }
            }

            word = new_word;
        }

        let result = word.join(" ");
        self.cache.borrow_mut().insert(token.to_string(), result.clone());
        result
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.encoder.len()
    }
}

impl Default for ClipTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Get all adjacent pairs in a word
fn get_pairs(word: &[String]) -> Vec<(String, String)> {
    word.windows(2)
        .map(|w| (w[0].clone(), w[1].clone()))
        .collect()
}

/// Build byte-to-unicode mapping
///
/// CLIP uses a specific mapping where printable ASCII maps to itself,
/// and other bytes map to Unicode private use area.
fn bytes_to_unicode() -> HashMap<u8, char> {
    let mut bs: Vec<u8> = Vec::new();

    // Printable ASCII ranges
    bs.extend(b'!'..=b'~');
    bs.extend(b'\xa1'..=b'\xac');
    bs.extend(b'\xae'..=b'\xff');

    let mut cs: Vec<char> = bs.iter().map(|&b| b as char).collect();

    // Map remaining bytes to private use area
    let mut n = 0u32;
    for b in 0u8..=255 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(char::from_u32(256 + n).unwrap());
            n += 1;
        }
    }

    bs.into_iter().zip(cs).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytes_to_unicode() {
        let mapping = bytes_to_unicode();
        assert_eq!(mapping.len(), 256);

        // Check printable ASCII maps to itself
        assert_eq!(mapping.get(&b'a'), Some(&'a'));
        assert_eq!(mapping.get(&b'Z'), Some(&'Z'));
        assert_eq!(mapping.get(&b'5'), Some(&'5'));
    }

    #[test]
    fn test_special_tokens() {
        assert_eq!(START_OF_TEXT, 49406);
        assert_eq!(END_OF_TEXT, 49407);
    }
}
