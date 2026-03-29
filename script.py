# Imports
import pandas as pd
import numpy as np
import urllib.request as libreq
from groq import Groq
import xml.etree.ElementTree as ET
import certifi
import os
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()
from bs4 import BeautifulSoup
import re
import PyPDF2
import io
import warnings
warnings.filterwarnings('ignore')

os.environ['SSL_CERT_FILE'] = certifi.where()

# Load the previous data
prev_df = pd.read_csv('datasets/arxiv_papers.csv')
prev_non_existent = pd.read_csv('datasets/non_existent.csv')
non_existent_dates = []
non_existent_titles = []
# Link to the arXiv page
link = 'https://arxiv.org/list/astro-ph/new'
page = libreq.urlopen(link)
html = page.read().decode('utf-8')

# Parse the HTML
soup = BeautifulSoup(html, 'html.parser')

# Check the data for arXiv papers
h3_tag = soup.find('h3', string=lambda x: x and 'Showing new listings for' in x)
if h3_tag:
    date_str = h3_tag.string.split('for ')[1]
    paper_date = datetime.strptime(date_str, '%A, %d %B %Y').strftime('%Y-%m-%d')
    print(f"Papers from date: {paper_date}")
else:
    print("Date tag not found")

# Check if the data is already present
length_prev_df = len(prev_df)
prev_date = prev_df['date'][length_prev_df - 1] if length_prev_df > 0 else None

if paper_date == prev_date:
    print('No new papers found')
    exit()

# Check the number of papers
h3_tag = soup.find('h3', string=lambda x: x and 'New submissions' in x)
if h3_tag:
    number_of_papers = int(h3_tag.string.split('(')[1].split()[1])
    print(f"Number of papers: {number_of_papers}")
else:
    print("Tag not found")

# Define the function to extract metadata
def extract_paper_metadata(xml_part):
    soup = BeautifulSoup(xml_part, 'html.parser')

    # Title
    title_tag = soup.find('div', class_='list-title mathjax')
    title = title_tag.get_text(strip=True).replace('Title:', '').strip() if title_tag else None

    # Abstract
    abstract_tag = soup.find('p', class_='mathjax')
    abstract = abstract_tag.get_text(strip=True) if abstract_tag else None

    # Authors
    authors_section = soup.find('div', class_='list-authors')
    authors = [author.get_text(strip=True) for author in authors_section.find_all('a')] if authors_section else []

    # Comments
    comments_tag = soup.find('div', class_='list-comments mathjax')
    comments = comments_tag.get_text(strip=True).replace('Comments:', '').strip() if comments_tag else ''
    
    # Figures, Pages, Tables
    figures_match = re.search(r'(\d+)\s+figures', comments)
    figures = int(figures_match.group(1)) if figures_match else None
    pages_match = re.search(r'(\d+)\s+pages', comments)
    pages = int(pages_match.group(1)) if pages_match else None
    tables_match = re.search(r'(\d+)\s+table[s]?', comments)
    tables = int(tables_match.group(1)) if tables_match else None

    # PDF link
    pdf_tag = soup.find('a', title='Download PDF')
    pdf_link = pdf_tag['href'] if pdf_tag else None

    # Primary Subject
    primary_subject_tag = soup.find('span', class_='primary-subject')
    primary_subject = primary_subject_tag.get_text(strip=True) if primary_subject_tag else None

    # Secondary Subjects
    secondary_subjects_section = soup.find('div', class_='list-subjects').get_text(strip=True)
    subjects_split = secondary_subjects_section.split(';')
    secondary_subjects = [subject.strip() for subject in subjects_split[1:]] if len(subjects_split) > 1 else None

    # Journal
    submitted_journal = comments.split('Submitted to ')[-1] if 'Submitted to' in comments else None
    submitted_journal = comments.split('Accepted to ')[-1] if 'Accepted to' in comments else submitted_journal
    submitted_journal = comments.split('Accepted for publication in ')[-1] if 'Accepted for publication in' in comments else submitted_journal
    submitted_journal = comments.split('Accepted by ')[-1] if 'Accepted by' in comments else submitted_journal
    submitted_journal = comments.split('Submitted by ')[-1] if 'Submitted by' in comments else submitted_journal

    # Published
    published_tag = soup.find('div', class_='list-journal-ref')
    published_journal = published_tag.get_text(strip=True).replace('Journal-ref:', '').strip() if published_tag else None

    return {
        'title': title,
        'abstract': abstract,
        'authors': authors,
        'figures': figures,
        'pages': pages,
        'tables': tables,
        'pdf_link': 'arxiv.org' + pdf_link,
        'primary_subject': primary_subject,
        'secondary_subjects': secondary_subjects,
        'submitted_journal': submitted_journal,
        'published_journal': published_journal
    }

# Function to extract all papers
def extract_all_papers(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    all_metadata = []

    # <a name='itemX'>
    items = soup.find_all('a', attrs={'name': True})

    for i in tqdm(range(number_of_papers - 1), desc="Extracting paper metadata"):
        start = items[i]
        end = items[i + 1]

        start_index = str(soup).find(str(start))
        end_index = str(soup).find(str(end))
        xml_part = str(soup)[start_index:end_index]

        metadata = extract_paper_metadata(xml_part)
        all_metadata.append(metadata)

    last_item = end
    start_index = str(soup).find(str(last_item))
    xml_part = str(soup)[start_index:]
    metadata = extract_paper_metadata(xml_part)
    all_metadata.append(metadata)

    return all_metadata

# Dataframe conversion function
def metadata_to_dataframe(metadata_list):
    return pd.DataFrame(metadata_list)

# Extract metadata
metadata_list = extract_all_papers(html)
df = metadata_to_dataframe(metadata_list)
print('Retrieved all Metadata')

# Remove brackets
def remove_brackets(text):
    return re.sub(r'\(.*?\)', '', text).strip()

# Data preprocessing
df['primary_subject'] = df['primary_subject'].map(remove_brackets)
df['secondary_subjects'] = df['secondary_subjects'].map(lambda x: [remove_brackets(subject) for subject in x], na_action='ignore') 
df['submitted_journal'] = df['submitted_journal'].str.split(r'[,;:.]').str[0]

# For loop to retrieve missing metadata
for i in tqdm(range(len(df)), desc="Retrieving missing metadata"):
    try:
        if pd.isna(df['pages'][i]) or pd.isna(df['figures'][i]) or pd.isna(df['tables'][i]):

            pdf_link = df['pdf_link'][i]
            pdf_response = libreq.urlopen('https://' + pdf_link)
            pdf_file = pdf_response.read()
            print(f"Processing PDF: {pdf_link}")
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file))

            # Number of Pages
            if pd.isna(df['pages'][i]):
                num_pages = len(pdf_reader.pages)
                df['pages'][i] = num_pages

            # Number of Figures
            if pd.isna(df['figures'][i]):
                highest_figure_number = 0
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    figure_numbers = re.findall(r'(?i)(?:Figure|Fig.|Figure.|Fig})\s+(\d+)', text)
                    if figure_numbers:
                        highest_figure_number = max(highest_figure_number, max(map(int, figure_numbers)))
                df['figures'][i] = highest_figure_number

            # Number of Tables
            if pd.isna(df['tables'][i]):
                highest_table_number = 0
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    table_numbers = re.findall(r'(?i)(?:Table|Table.})\s+(\d+)', text)
                    if table_numbers:
                        highest_table_number = max(highest_table_number, max(map(int, table_numbers)))
                df['tables'][i] = highest_table_number
    except:
        non_existent_dates.append(paper_date)
        non_existent_titles.append(df['title'][i])
        print(f"Metadata for Paper: {df['title'][i]}    doesn't exist")

non_existent = pd.DataFrame({'date': non_existent_dates, 'title': non_existent_titles})
non_existent_write = pd.concat([prev_non_existent, non_existent], ignore_index=True)
non_existent_write.to_csv('non_existent.csv')

print('Retrieved missing Metadata')

# For loop to retrieve keywords
df['keywords'] = None  

for i in tqdm(range(len(df)), desc="Retrieving keywords"):
    try:
        links = df['pdf_link'][i]
        pdf_response = libreq.urlopen('https://' + links)
        pdf_file = pdf_response.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file))

        keywords = []
        for page in pdf_reader.pages:
            text = page.extract_text()
            text = re.sub(r'\s+', ' ', text) 

            patterns = [
                r'(?i)(?:keyword[s]?|Uniﬁed Astronomy Thesaurus concepts?|key words?|Key words?|Subject headings)[:.]?\s*(.*?)\s*(?=(?:[.;]|\n|$))'
            ]

            for pattern in patterns:
                matches = re.findall(pattern, text, re.DOTALL)
                for match in matches:

                    split_keywords = re.split(r'[;,\n]', match)
                    keywords.extend([kw.strip() for kw in split_keywords if kw.strip()])

        keywords = list(set(keywords))

        stop_phrases = ['1. Introduction', '1 Introduction']
        for stop_phrase in stop_phrases:
            if any(stop_phrase in keyword for keyword in keywords):
                keywords = ' '.join(keywords).split(stop_phrase)[0]
                break

        df.at[i, 'keywords'] = keywords
    except:
        print(f"cannot retrieve keywords for paper {df['title'][i]}")
print('Retrieved Keywords')

# Save the date
df['date'] = paper_date

# Save the data
new_df = pd.concat([prev_df, df], ignore_index=True)
new_df.to_csv('arxiv_papers.csv', index=False)
