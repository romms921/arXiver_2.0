import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import os
import re
import time
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
CSV_PATH = 'datasets/arxiv_papers.csv'
LOG_PATH = 'log.txt'
BASE_URL = "https://arxiv.org"
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

def get_start_date():
    """Determines the starting date based on existing data or current year."""
    if os.path.exists(CSV_PATH):
        try:
            temp_df = pd.read_csv(CSV_PATH)
            if not temp_df.empty and 'date' in temp_df.columns:
                last_date_str = temp_df['date'].iloc[-1]
                # Format: DD-MM-YYYY
                last_date = datetime.strptime(last_date_str, '%d-%m-%Y')
                start_date = last_date + timedelta(days=1)
                print(f"Resuming from: {start_date.strftime('%Y-%m-%d')}")
                return start_date.strftime('%Y-%m-%d')
        except Exception as e:
            print(f"Error reading CSV: {e}")
    
    current_year = datetime.now().year
    start_date = f"{current_year}-01-01"
    print(f"No existing data. Starting from: {start_date}")
    return start_date

def log_progress(day_name, date_str, count):
    """Appends info to log.txt: Thursday 01 Jan 2026 - 58"""
    with open(LOG_PATH, 'a') as f:
        f.write(f"{day_name} {date_str} - {count}\n")

def parse_paper_metadata(dt_tag, dd_tag, index):
    """Extracts metadata from the <dt> and <dd> tags pair."""
    
    # 1. PDF Link: Look for anchor inside <dt>
    pdf_anchor = dt_tag.find('a', title='Download PDF')
    if pdf_anchor and 'href' in pdf_anchor.attrs:
        pdf_link = "arxiv.org" + pdf_anchor['href']
    else:
        # Fallback to ID-based construction
        abs_anchor = dt_tag.find('a', title='Abstract')
        p_id = abs_anchor.get('id') if abs_anchor else None
        pdf_link = f"arxiv.org/pdf/{p_id}" if p_id else "N/A"

    # 2. Title
    title_tag = dd_tag.find('div', class_='list-title')
    title = title_tag.get_text(strip=True).replace('Title:', '').strip() if title_tag else "N/A"

    # 3. Abstract
    abstract_tag = dd_tag.find('p', class_='mathjax')
    abstract = abstract_tag.get_text(strip=True) if abstract_tag else "N/A"

    # 4. Authors
    authors_div = dd_tag.find('div', class_='list-authors')
    authors = [a.get_text(strip=True) for a in authors_div.find_all('a')] if authors_div else []

    # 5. Comments
    comments_div = dd_tag.find('div', class_='list-comments')
    comments = comments_div.get_text(strip=True).replace('Comments:', '').strip() if comments_div else ""

    # 6. Journal Ref
    jref_div = dd_tag.find('div', class_='list-journal-ref')
    journal_ref = jref_div.get_text(strip=True).replace('Journal-ref:', '').strip() if jref_div else None

    # 7. Subjects
    subjects_div = dd_tag.find('div', class_='list-subjects')
    primary_subject = ""
    secondary_subjects = []
    if subjects_div:
        text = subjects_div.get_text(strip=True).replace('Subjects:', '')
        parts = text.split(';')
        primary_subject = re.sub(r'\(.*?\)', '', parts[0]).strip()
        if len(parts) > 1:
            secondary_subjects = [re.sub(r'\(.*?\)', '', s).strip() for s in parts[1:]]

    return {
        'daily_Index': index,
        'title': title,
        'abstract': abstract,
        'authors': authors,
        'pdf_link': pdf_link,
        'comments': comments,
        'journal_ref': journal_ref,
        'primary_subject': primary_subject,
        'secondary_subjects': secondary_subjects
    }

def scrape_arxiv():
    start_date_str = get_start_date()
    # Initial Catchup URL
    current_url = f"https://arxiv.org/catchup/astro-ph/{start_date_str}?abs=True"
    
    if not os.path.exists('datasets'):
        os.makedirs('datasets')

    while current_url:
        print(f"\n--- Loading: {current_url} ---")
        try:
            response = requests.get(current_url, headers=HEADERS)
            response.raise_for_status()
        except Exception as e:
            print(f"Request Error: {e}")
            break

        soup = BeautifulSoup(response.text, 'html.parser')

        # 1. Day and Date Extraction (XPATH: //*[@id="dlpage"]/h1)
        h1_tag = soup.find('h1')
        if not h1_tag or "Catchup results" not in h1_tag.text:
            print("No catchup header found. Ending sequence.")
            break
            
        header_text = h1_tag.get_text()
        date_match = re.search(r'on (\w+), (\d{2} \w+ \d{4})', header_text)
        if not date_match:
            print("Could not parse date from H1.")
            break
            
        weekday = date_match.group(1)
        raw_date_str = date_match.group(2)
        formatted_date = datetime.strptime(raw_date_str, '%d %b %Y').strftime('%d-%m-%Y')

        # 2. Submission Count (XPATH: //*[@id="articles"]/h3)
        h3_new = soup.find('h3', string=lambda x: x and 'New submissions' in x)
        expected_count = 0
        if h3_new:
            count_match = re.search(r'(\d+)\s+entries', h3_new.text)
            if count_match:
                expected_count = int(count_match.group(1))

        if expected_count == 0:
            print(f"No new papers for {formatted_date}.")
            log_progress(weekday, raw_date_str, 0)
        else:
            # 3. Extraction logic
            all_dt = soup.find_all('dt')
            all_dd = soup.find_all('dd')
            actual_to_scrape = min(len(all_dt), len(all_dd), expected_count)
            
            daily_batch = []
            for i in tqdm(range(actual_to_scrape), desc=f"Day: {formatted_date}"):
                paper_meta = parse_paper_metadata(all_dt[i], all_dd[i], i + 1)
                paper_meta['date'] = formatted_date
                paper_meta['weekday'] = weekday
                daily_batch.append(paper_meta)

            # 4. Periodic Save
            if daily_batch:
                new_df = pd.DataFrame(daily_batch)
                if os.path.exists(CSV_PATH):
                    new_df.to_csv(CSV_PATH, mode='a', header=False, index=False)
                else:
                    new_df.to_csv(CSV_PATH, index=False)
                
                log_progress(weekday, raw_date_str, len(daily_batch))
                print(f"Saved {len(daily_batch)} papers.")

        # 5. FIND NEXT PAGE (Forward in time)
        # Based on XPATH //*[@id="dlpage"]/div[1]/a, we look inside the first div of dlpage
        # We look for the link that specifically says "Next" to avoid going backwards via "Prev"
        dlpage = soup.find(id='dlpage')
        next_link_tag = None
        
        if dlpage:
            # Find the first pagination div
            pagination_div = dlpage.find('div', class_='list-pagination-top')
            if not pagination_div:
                # Fallback to the first div if class is different
                pagination_div = dlpage.find('div')
            
            if pagination_div:
                # Find the link that contains "Next" (case insensitive)
                next_link_tag = pagination_div.find('a', string=re.compile(r'Next', re.I))

        if next_link_tag and 'href' in next_link_tag.attrs:
            current_url = BASE_URL + next_link_tag['href']
            print("Rate limit wait: 15 seconds...")
            time.sleep(15)
        else:
            print("No 'Next' link found. Scraping sequence complete.")
            current_url = None

    print("\n--- Process Finished ---")

if __name__ == "__main__":
    scrape_arxiv()