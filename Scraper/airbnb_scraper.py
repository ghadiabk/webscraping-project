import time
import random
import re
import json
import threading
from urllib.parse import urlparse

import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from concurrent.futures import ThreadPoolExecutor, as_completed


# ================== CONFIG ==================
START_URL = "https://www.airbnb.com/s/Lebanon/homes"

MAX_PAGES = 20
MAX_LISTINGS = 500
MAX_THREADS = 6
HEADLESS = True

OUTPUT_CSV = (
    r"F:\University\Semester 5 - Fall 2025\Data Science and Web Scraping"
    r"\Course_Project\webscraping-project\airbnb_lebanon_full.csv"
)

DELAY_MIN, DELAY_MAX = 1.5, 3.0
# ===========================================


COLUMNS = [
    "platform",
    "listing_id",
    "title",
    "city",
    "bedrooms",
    "bathrooms",
    "beds",
    "price",
    "rating",
    "review_count",
    "amenities_count",
    "minimum_nights",
    "last_scraped",
]


def sleep():
    time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))


def clean(s):
    if not s:
        return ""
    return (
        s.replace("\u2009", " ")
         .replace("\u202f", " ")
         .replace("\xa0", " ")
         .strip()
    )


def setup_driver(headless=HEADLESS):
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--disable-infobars")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--lang=en-US")

    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=opts)


# ============================================================
#                 CARD PARSER  (working version)
# ============================================================

def parse_card_text(text):
    text = clean(text)
    lines = [clean(l) for l in text.split("\n") if clean(l)]

    title = None
    bedrooms = None
    beds = None
    price = None
    min_nights = None
    review_count = None

    for line in lines:
        lower = line.lower()

        # ---------------- TITLE ----------------
        if (
            not title
            and not any(x in lower for x in [
                "bed", "bath", "night", "review", "$",
                "superhost", "·", "hosted", "stay", "for"
            ])
            and not re.search(r"\d{1,2}\s*[a-zA-Z]{3}", line)
        ):
            if len(line.strip()) > 3:
                title = line
                continue

        # ---------------- BEDROOMS ----------------
        if "bedroom" in lower:
            m = re.search(r"(\d+)\s*bedroom", lower)
            if m:
                bedrooms = int(m.group(1))

        # ---------------- BEDS ----------------
        if "bed" in lower:
            m = re.search(r"(\d+)\s*beds?", lower)
            if m:
                beds = int(m.group(1))

        # ---------------- PRICE + MIN NIGHTS ----------------
        if "night" in lower:
            m_nights = re.search(r"(\d+)\s*nights?", lower)
            if m_nights:
                min_nights = int(m_nights.group(1))

        if "$" in line:
            m_price = re.search(r"\$(\d[\d,]*)", line)
            if m_price:
                price = int(m_price.group(1).replace(",", ""))

        # ---------------- REVIEW COUNT ----------------
        m = re.search(r"\((\d+)\)", line)  # (270)
        if m:
            review_count = int(m.group(1))
            continue

        m = re.search(r"(\d+)\s*reviews?", lower)  # 270 reviews
        if m:
            review_count = int(m.group(1))
            continue

    return {
        "title": title,
        "bedrooms": bedrooms,
        "beds": beds,
        "price": price,
        "minimum_nights": min_nights,
        "review_count": review_count,
        "bathrooms": None,
        "amenities_count": None,
    }


# ============================================================
#           JSON-LD + LISTING-LEVEL HELPERS
# ============================================================

def extract_jsonld(driver):
    try:
        script = driver.find_element(By.XPATH, "//script[@type='application/ld+json']")
        return json.loads(script.get_attribute("innerHTML"))
    except Exception:
        return None


# ============================================================
#     UNIVERSAL BATHROOM EXTRACTOR (ALWAYS WORKS)
# ============================================================

def extract_bathrooms_anywhere(driver):
    try:
        html = driver.page_source.lower()

        html = html.replace("\u202f"," ").replace("\u2009"," ")

        patterns = [
            r"(\d+(\.\d+)?)\s*bath",
            r"(\d+(\.\d+)?)\s*private bath",
            r"shared bath",
        ]

        for p in patterns:
            m = re.search(p, html)
            if m:
                if m.group(1):
                    return float(m.group(1))
                return 1
    except:
        return None

    return None


# ============================================================
#               AMENITIES COUNT
# ============================================================

def parse_amenities_count(driver):
    count = None
    try:
        driver.execute_script("window.scrollTo(0, 1500);")
    except:
        pass
    sleep()

    try:
        elems = driver.find_elements(
            By.XPATH,
            "//*[contains(translate(., 'AMENITIES', 'amenities'), 'amenities')]"
        )

        for el in elems:
            txt = clean(el.get_attribute("textContent") or "").lower()
            m = re.search(r"(\d+)\s+amenities", txt)
            if m:
                count = int(m.group(1))
                break
    except:
        pass

    return count


def parse_rating_and_reviews_from_jsonld(j):
    rating = None
    review_count = None
    if j and isinstance(j, dict) and "aggregateRating" in j:
        agg = j["aggregateRating"]
        if isinstance(agg, dict):
            rating = agg.get("ratingValue")
            review_count = agg.get("reviewCount")
    return rating, review_count


# ============================================================
#       COLLECT CARD LINKS + CARD FEATURES (PAGINATION)
# ============================================================

def collect_cards_with_pagination():
    driver = setup_driver(headless=HEADLESS)
    driver.get(START_URL)
    sleep()

    cards_map = {}

    for page in range(1, MAX_PAGES + 1):
        print("Page", page, "- loading cards...")
        for _ in range(5):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            sleep()

        cards = driver.find_elements(By.XPATH, "//div[@itemprop='itemListElement']")
        print("  Found", len(cards), "cards")

        for card in cards:
            try:
                a = card.find_element(By.TAG_NAME, "a")
                href = a.get_attribute("href") or ""
                if "/rooms/" not in href:
                    continue

                parsed = urlparse(href)
                base_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

                if base_url in cards_map:
                    continue

                text = clean(card.text)
                card_data = parse_card_text(text)
                cards_map[base_url] = card_data

                if len(cards_map) >= MAX_LISTINGS:
                    break

            except:
                continue

        print("  Total:", len(cards_map))

        if len(cards_map) >= MAX_LISTINGS:
            break

        try:
            next_btn = driver.find_element(
                By.XPATH, "//a[@aria-label='Next' or contains(.,'Next')]"
            )
            driver.execute_script("arguments[0].click();", next_btn)
            sleep()
        except NoSuchElementException:
            print("No further pages.")
            break

    driver.quit()
    print("Collected", len(cards_map), "unique listings.")
    return cards_map


# ============================================================
#                LISTING SCRAPER (DETAIL PAGE)
# ============================================================

def scrape_listing(url, card_data):
    driver = setup_driver(headless=HEADLESS)

    res = {c: None for c in COLUMNS}
    res["platform"] = "airbnb"
    res["title"] = card_data.get("title")
    res["bedrooms"] = card_data.get("bedrooms")
    res["beds"] = card_data.get("beds")
    res["price"] = card_data.get("price")
    res["minimum_nights"] = card_data.get("minimum_nights")
    res["review_count"] = card_data.get("review_count")
    res["last_scraped"] = time.strftime("%Y-%m-%d %H:%M:%S")

    try:
        parts = url.rstrip("/").split("/")
        if "rooms" in parts:
            res["listing_id"] = parts[parts.index("rooms") + 1]
    except:
        pass

    try:
        driver.get(url)
        sleep()

        j = extract_jsonld(driver)

        # city
        if j and "address" in j and isinstance(j["address"], dict):
            addr = j["address"]
            res["city"] = (
                addr.get("addressLocality")
                or addr.get("addressRegion")
                or addr.get("streetAddress")
            )

        # bedrooms override
        if res["bedrooms"] is None and j:
            res["bedrooms"] = (
                j.get("numberOfRooms")
                or j.get("numberOfBedrooms")
            )

        # bathrooms (always use the universal extractor)
        res["bathrooms"] = extract_bathrooms_anywhere(driver)

        # rating + reviews
        rating_json, reviews_json = parse_rating_and_reviews_from_jsonld(j)
        if rating_json is not None:
            try:
                res["rating"] = float(rating_json)
            except:
                res["rating"] = rating_json

        if res["review_count"] is None and reviews_json is not None:
            try:
                res["review_count"] = int(reviews_json)
            except:
                res["review_count"] = reviews_json

        # amenities count
        res["amenities_count"] = parse_amenities_count(driver)

        # numeric cleanup
        for f in ["bedrooms", "bathrooms", "beds", "review_count", "amenities_count"]:
            try:
                if res[f] is not None:
                    res[f] = int(res[f])
            except:
                pass

        if res["rating"] is not None:
            try:
                res["rating"] = float(res["rating"])
            except:
                pass

    except Exception as e:
        print("[Thread error]", url, ":", e)

    finally:
        try:
            driver.quit()
        except:
            pass

    return res


def scrape_worker(url, card_data):
    return scrape_listing(url, card_data)


# ============================================================
#                           MAIN
# ============================================================

def main():
    cards_map = collect_cards_with_pagination()
    if not cards_map:
        print("No listings found.")
        return

    urls = list(cards_map.keys())
    print("Scraping", len(urls), "listings with", MAX_THREADS, "threads.")

    rows = []
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as ex:
        futures = {ex.submit(scrape_worker, u, cards_map[u]): u for u in urls}

        for fut in as_completed(futures):
            u = futures[fut]
            try:
                data = fut.result()
                if data:
                    with lock:
                        rows.append(data)
                print("[OK]", u)
            except Exception as e:
                print("[Error]", u, ":", e)

    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print("Saved", len(rows), "listings to", OUTPUT_CSV)


if __name__ == "__main__":
    main()
