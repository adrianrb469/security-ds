# Lab 5: Security Data Science - DGA Detection

## Overview

This project focuses on identifying potentially malicious Top-Level Domains (TLDs) often associated with Domain Generation Algorithms (DGAs) within a dataset of DNS events. The process involves several stages: data preprocessing, TLD classification using a large language model (Gemini), and validation using a domain whitelist and WHOIS information.

## Files

- **`preprocessing.ipynb`**:
  - Loads DNS event data from `data/events.json`.
  - Filters for DNS event types.
  - Normalizes the nested JSON structure.
  - Filters for DNS 'A' records.
  - Extracts unique domain names (`dns_rrname`).
  - Applies a custom function `get_tld` to extract the effective TLD for each unique domain.
  - Saves the unique TLDs to `data/unique_tlds.csv`.
- **`gemini.ipynb`**:
  - Loads the unique TLDs.
  - Uses the Google Gemini API (`gemini-1.5-flash-002`) to classify each TLD.
  - The prompt asks the model to classify a TLD as DGA-generated (1) or legitimate (0).
  - Saves the TLDs along with their classification to `data/tld_dga_classification.csv`.
- **`validation.ipynb`**:
  - Loads the Gemini-classified TLDs (`data/tld_dga_classification.csv`).
  - Loads a domain whitelist (`data/top_tld.csv`).
  - Identifies TLDs classified as DGA (1) by Gemini.
  - Filters the DGA list further by checking if the TLD appears as a substring within any domain in the whitelist. TLDs _not_ found in the whitelist are considered "highly suspicious".
  - Uses the `python-whois` library to attempt to retrieve the creation date for these highly suspicious TLDs.
  - Outputs the final list of highly suspicious TLDs, their classification, and their retrieved creation date (if available).
- **`data/`**: Directory containing input data and intermediate/output CSV files.
  - `events.json` Original DNS event data.
  - `top_tld.csv`: Whitelist of known legitimate domains.
  - `unique_tlds.csv`: Output from `preprocessing.ipynb`.
  - `tld_dga_classification.csv`: Output from `gemini.ipynb`.

## Analysis

Analysis and discussion of the results can be found in `lab5.pdf`
