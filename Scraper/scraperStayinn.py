from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import os
import csv
import time

BASE_URL = "https://www.stayinn.com/listing-search-results"
FIELDNAMES = [
    "platform",
    "listing_id",
    "title",
    "city",
    "bedrooms",
    "bathrooms",
    "price_per_night_usd",
    "rating",
    "review_count",
    "amenities_count",
    "url",
    "last_scraped",
]

def get_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    chrome_bin = os.environ.get('CHROME_BIN') or os.environ.get('CHROME_PATH')
    if chrome_bin:
        try:
            chrome_options.binary_location = chrome_bin
        except Exception:
            pass

    chrome_driver_path = os.environ.get('CHROME_DRIVER')
    if chrome_driver_path:
        return webdriver.Chrome(
            service=Service(executable_path=chrome_driver_path),
            options=chrome_options
        )
    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )

def scrape_listing_detail(url):
    driver = get_driver()
    try:
        driver.get(url)
        WebDriverWait(driver, 8).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
    except Exception:
        pass

    data = scrape_details_from_current_page(driver, url)
    driver.quit()
    return data


def scrape_details_from_current_page(driver, url=None):
    try:
        wait = WebDriverWait(driver, 6)

        def safe_find_text(by, selector):
            try:
                el = driver.find_element(by, selector)
                return el.text.strip()
            except Exception:
                return None

        title = safe_find_text(By.ID, 'listing-title') or safe_find_text(By.CSS_SELECTOR, 'h1')
        city = safe_find_text(By.XPATH, '/html/body/div[1]/div[3]/div/div/div/div[4]/div[1]/div[1]/h2')

        bedrooms = None
        bathrooms = None
        try:
            info_spans = driver.find_elements(By.XPATH, "//section//span[normalize-space()]")
            for s in info_spans:
                t = s.text.lower()
                if 'bed' in t and bedrooms is None:
                    bedrooms = s.text.strip()
                if 'bath' in t and bathrooms is None:
                    bathrooms = s.text.strip()
                if bedrooms and bathrooms:
                    break
        except Exception:
            pass
    
        price_per_night_usd = None
        try:
            price_el = WebDriverWait(driver, 4).until(
                EC.presence_of_element_located((By.XPATH, "/html/body/div[1]/div[3]/div/div/div/div[4]/div[2]/div/div[2]/div/div/div/div/div/p[2]"))
            )
            price_per_night_usd = price_el.text.strip()
        except Exception:
            price_per_night_usd = None

        rating = None
        review_count = None
        try:
            try:
                rating_el = WebDriverWait(driver, 3).until(
                    EC.presence_of_element_located((By.XPATH, "/html/body/div[1]/div[3]/div/div/div/div[4]/div[1]/div[1]/aside/div/span"))
                )
                rating = rating_el.text.strip()
            except Exception:
                rating = safe_find_text(By.XPATH, "//span[contains(@class,'rating') or contains(@class,'score')]")

            review_count = safe_find_text(By.XPATH, "//a[contains(@href,'#reviews') or contains(text(),'review')]")
        except Exception:
            pass

        # Extract amenities count
        amenities_count = None
        try:
            amenities_el = driver.find_element(By.XPATH, "/html/body/div[1]/div[3]/div/div/div/div[4]/div[1]/section[2]/button")
            amenities_count = amenities_el.text.strip()
        except Exception:
            amenities_count = None
            
        listing_id = None
        if url:
            listing_id = url.rstrip('/').split('/')[-1]

        return {
            "platform": "Stayinn",
            "listing_id": listing_id,
            "title": title,
            "city": city,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "price_per_night_usd": price_per_night_usd,
            "rating": rating,
            "review_count": review_count,
            "amenities_count": amenities_count,
            "url": url or driver.current_url,
            "last_scraped": datetime.utcnow().isoformat()
        }

    except Exception as e:
        print('scrape error:', e)
        return None

def get_listing_urls(max_pages=20):
    driver = get_driver()
    all_urls = []

    for page in range(1, max_pages + 1):
        print(f"[INFO] Loading search page {page}")
        driver.get(f"{BASE_URL}?page={page}")
        time.sleep(2)

        try:
            anchors = driver.find_elements(By.CSS_SELECTOR, "a[data-testid='listing-card']")
            for a in anchors:
                try:
                    href = a.get_attribute('href')
                    if href and href not in all_urls:
                        all_urls.append(href)
                except StaleElementReferenceException:
                    continue
        except Exception:
            continue

    driver.quit()
    return all_urls

def scrape_all_listings(max_pages=1, workers=5):
    urls = get_listing_urls(max_pages)
    print(f"[INFO] Found {len(urls)} listings across {max_pages} pages.")

    results = []
    if not urls:
        return results

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(scrape_listing_detail, url): url for url in urls}

        for future in as_completed(futures):
            url = futures[future]
            try:
                data = future.result()
                if data:
                    results.append(data)
            except Exception as e:
                print(f"[ERROR] Scrape failed for {url}: {e}")

    # Assign incremental listing_id values deterministically
    for idx, item in enumerate(results, start=1):
        item['listing_id'] = idx

    return results

def save_csv(data):
    if not data:
        print("No data scraped.")
        return

    file_path = "stayinn_listings.csv"
    write_header = not os.path.exists(file_path)

    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        # Normalize rows to fieldnames order and provide defaults for missing keys
        rows = []
        for d in data:
            row = {k: d.get(k, "") for k in FIELDNAMES}
            rows.append(row)

        writer.writerows(rows)

    action = "Created" if write_header else "Appended"
    print(f"[DONE] {action} {len(data)} rows to {file_path}")

if __name__ == "__main__":
    # Allow overriding from environment (useful for CI)
    try:
        max_pages = int(os.environ.get('MAX_PAGES', '10'))
    except Exception:
        max_pages = 10
    try:
        workers = int(os.environ.get('WORKERS', '5'))
    except Exception:
        workers = 5

    data = scrape_all_listings(max_pages=max_pages, workers=workers)
    save_csv(data)