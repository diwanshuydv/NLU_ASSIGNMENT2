"""
===============================================================================
scraper.py — Data Collection for IIT Jodhpur Word2Vec Training
===============================================================================
This script collects textual data from multiple IIT Jodhpur sources:
  1. Academic Regulations PDF (extracted via pdfplumber)
  2. Official IIT Jodhpur website pages (departments, about, programs, etc.)
  3. Faculty profile pages

The scraped text is saved as individual .txt files to data/raw/ for downstream
preprocessing. We use requests + BeautifulSoup for web scraping and pdfplumber
for PDF text extraction.
===============================================================================
"""

import os
import re
import time
import requests
from bs4 import BeautifulSoup
import pdfplumber

# --------------------------------------------------------------------------
# Configuration: Output directory and list of URLs to scrape
# --------------------------------------------------------------------------
RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "Academic_Regulations_Final_03_09_2019.pdf")

# IIT Jodhpur web pages to scrape — covering departments, academic programs,
# research, about page, and announcements as required by the assignment.
URLS = [
    # About & Overview pages
    "https://iitj.ac.in/",
    "https://iitj.ac.in/main/en/introduction",
    "https://iitj.ac.in/main/en/history",
    
    # Academic Pages
    "https://iitj.ac.in/office-of-academics/en/academics",
    "https://iitj.ac.in/Bachelor-of-Technology/en/Bachelor-of-Technology",
    "https://iitj.ac.in/mechanical-engineering/en/undergraduate-program",
    "https://iitj.ac.in/electrical-engineering/en/undergraduate-program",
    # Department pages — CSE, EE, ME, Math, Physics, Chemistry, etc.
    "https://cse.iitj.ac.in/",
    "https://ee.iitj.ac.in/",
    "https://iitj.ac.in/bioscience-bioengineering",
    "https://iitj.ac.in/mathematics/",
    "https://iitj.ac.in/physics/",
    "https://iitj.ac.in/chemistry/",
    "https://iitj.ac.in/civil-and-infrastructure-engineering/",
    #Talks
    "https://www.iitj.ac.in/Computer-Science-Engineering/en/Previous-Talks",
    # Research
    "https://iitj.ac.in/office-of-research-development/en/office-of-research-and-development",
    "https://iitj.ac.in/aiot-fab-facility/en/aiot-fab-facility?",
    "https://iitj.ac.in/crf/en/crf",
    "https://iitj.ac.in/crf/en/publications",
    "https://www.iitj.ac.in/computer-science-engineering/en/Research-Archive",
    "https://www.iitj.ac.in/computer-science-engineering/en/projects",
    # Admissions
    "https://www.iitj.ac.in/schools/en/mission-vision",
    "https://iitj.ac.in/computer-science-engineering/en/doctoral-programs",
    "https://iitj.ac.in/computer-science-engineering/en/master-of-technology",
    "https://iitj.ac.in/winter-school/en/Important-Information",
    # Student life & campus
    "https://iitj.ac.in/office-of-director/en/office-of-director",
    "https://iitj.ac.in/office-of-director/en/about-director",

    # Faculty pages
    "https://iitj.ac.in/People/List?dept=Mechanical-Engineering&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd",
    "https://iitj.ac.in/People/Profile/e9793af3-6d4f-4560-8489-2c256912a72a",
    "https://iitj.ac.in/People/List?dept=Computer-Science-Engineering&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd",
    "https://iitj.ac.in/People/List?dept=Mathematics&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd",
    "https://iitj.ac.in/People/List?dept=Physics&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd",
    "https://iitj.ac.in/People/List?dept=Chemistry&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd",
    "https://iitj.ac.in/People/Profile/c7aa696c-0f3d-48aa-ad32-eeb189bdca60",
]

# Request headers to mimic a real browser and avoid blocks
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def extract_pdf_text(pdf_path: str) -> str:
    """
    Extracts all text content from a PDF file using pdfplumber.
    
    Args:
        pdf_path: Absolute or relative path to the PDF file.
    
    Returns:
        Concatenated text from all pages of the PDF.
    """
    print(f"[PDF] Extracting text from: {pdf_path}")
    all_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                all_text.append(text)
    combined = "\n".join(all_text)
    print(f"  -> Extracted {len(combined)} characters from {len(all_text)} pages")
    return combined


def scrape_webpage(url: str) -> str:
    """
    Scrapes visible text content from a webpage using requests + BeautifulSoup.
    
    We remove script/style/nav/footer/header elements to reduce boilerplate,
    and extract text from the main content areas (paragraphs, headings, lists).
    
    Args:
        url: The URL of the webpage to scrape.
    
    Returns:
        Cleaned text content from the page, or empty string on failure.
    """
    try:
        print(f"[WEB] Scraping: {url}")
        resp = requests.get(url, headers=HEADERS, timeout=15,  verify=False)
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Remove non-content elements (scripts, styles, navbars, footers)
        for tag in soup.find_all(["script", "style", "nav", "footer", "header",
                                   "noscript", "meta", "link", "iframe"]):
            tag.decompose()
        
        # Extract text from content-bearing tags
        text_parts = []
        for tag in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6",
                                   "li", "td", "th", "span", "div", "a"]):
            text = tag.get_text(separator=" ", strip=True)
            if text and len(text) > 5:  # skip very short fragments
                text_parts.append(text)
        
        combined = "\n".join(text_parts)
        print(f"  -> Got {len(combined)} characters")
        return combined
        
    except Exception as e:
        print(f"  -> ERROR scraping {url}: {e}")
        return ""


def save_text(text: str, filename: str) -> None:
    """
    Saves text content to a file in the raw data directory.
    
    Args:
        text: The text content to save.
        filename: Name of the output file (will be placed in RAW_DIR).
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    filepath = os.path.join(RAW_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  -> Saved to {filepath}")


def main():
    """
    Main entry point: extracts text from the academic regulations PDF,
    then scrapes all configured IIT Jodhpur web pages. Each source is
    saved as a separate .txt file in data/raw/.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    
    # ---- Step 1: Extract text from the Academic Regulations PDF ----
    # This is a required source per the assignment instructions
    if os.path.exists(PDF_PATH):
        pdf_text = extract_pdf_text(PDF_PATH)
        save_text(pdf_text, "academic_regulations.txt")
    else:
        print(f"[WARN] PDF not found at {PDF_PATH}")
    
    # ---- Step 2: Scrape IIT Jodhpur web pages ----
    # We iterate through all configured URLs, scraping each one
    # with a polite delay between requests to avoid overloading the server
    for i, url in enumerate(URLS):
        text = scrape_webpage(url)
        if text:
            # Create a clean filename from the URL
            # e.g., "https://iitj.ac.in/about/index.php?id=overview" -> "iitj_about_overview.txt"
            clean_name = url.replace("https://", "").replace("http://", "")
            clean_name = re.sub(r"[^a-zA-Z0-9]", "_", clean_name)
            clean_name = re.sub(r"_+", "_", clean_name).strip("_")
            save_text(text, f"{clean_name}.txt")
        
        # Polite delay between requests (1 second)
        if i < len(URLS) - 1:
            time.sleep(1)
    
    print(f"\n[DONE] All data saved to {RAW_DIR}")
    print(f"  Total files: {len(os.listdir(RAW_DIR))}")


if __name__ == "__main__":
    main()
