import csv
import time
import re
from datetime import datetime
from urllib.parse impo 53t5ertrt urlparse

import requests
from lxml import html

BASE_URL = "https://lebanon-rental.com/properties-for-rent/"

# ✅ Custom User-Agent (as your doctor asked)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

# Optional constant LBP → USD (you can adjust or set to None)
LBP_TO_USD_RATE = 90000  # just an example for the project


def get_tree(url: str) -> html.HtmlElement:
    """Fetch a URL and return an lxml HTML tree."""
    print(f"[GET] {url}")
    resp = requests.get(url, headers=HEADERS, timeout=25)
    resp.raise_for_status()
    return html.fromstring(resp.text)


def extract_listing_id_from_url(url: str) -> str:
    """
    Use the last path segment of the property URL as listing_id.
    Example: /property/luxury-windmill-2-private-pool/ -> luxury-windmill-2-private-pool
    """
    path = urlparse(url).path
    segments = [seg for seg in path.split("/") if seg]
    return segments[-1] if segments else url


def parse_price_to_usd(raw_price: str):
    """
    Convert raw price string into a float in USD, if possible.
    Handles USD and (optionally) LBP.
    """
    if not raw_price:
        return None

    text = raw_price.replace("\xa0", " ").strip()

    # Does it clearly look like USD?
    usd_symbol = "$" in text or "USD" in text.upper()

    # Extract first number
    m = re.search(r"([\d.,]+)", text)
    if not m:
        return None

    num_str = m.group(1).replace(",", "")
    try:
        value = float(num_str)
    except ValueError:
        return None

    if usd_symbol:
        return value

    # Try to detect LBP and convert
    if LBP_TO_USD_RATE and (
        "LBP" in text.upper() or "ل.ل" in text or "LL" in text
    ):
        return round(value / LBP_TO_USD_RATE, 2)

    # Unknown currency → None (or treat as USD if you want)
    return None


def parse_card(card) -> dict:
    """
    Parse one property card (the big box you sent) into our schema.
    We stay entirely on the listing page — no extra HTTP requests.
    """

    # URL: from title link (avoids WhatsApp/Call links)
    url_nodes = card.xpath(
        './/h2[contains(@class,"item-title")]//a[@href][1]/@href'
    )
    url = url_nodes[0] if url_nodes else None

    # Title
    title_nodes = card.xpath(
        './/h2[contains(@class,"item-title")]//a/text()'
    )
    title = title_nodes[0].strip() if title_nodes else None

    # Address -> we’ll use last part after comma as "city"
    addr_nodes = card.xpath(
        './/address[contains(@class,"item-address")]//span/text()'
    )
    address = addr_nodes[0].strip() if addr_nodes else None
    city = None
    if address:
        parts = [p.strip() for p in address.split(",") if p.strip()]
        if parts:
            city = parts[-1]  # e.g. "Jbeil" from "Mount Lebanon, Jbeil"

    # Price text on card ("Started 220$/Night")
    price_nodes = card.xpath(
        './/ul[contains(@class,"item-price-wrap")]'
        '//li[contains(@class,"item-price")]//text()'
    )
    raw_price = (
        " ".join(p.strip() for p in price_nodes if p.strip())
        if price_nodes
        else None
    )
    price_per_night_usd = parse_price_to_usd(raw_price)

    # Amenities line in header (Bed, Bath icons)
    amenities_ul = card.xpath(
        './/ul[contains(@class,"item-amenities") and '
        'contains(@class,"item-amenities-with-icons")][1]'
    )
    beds = bedrooms = bathrooms = None
    amenities_count = 0

    if amenities_ul:
        ul = amenities_ul[0]
        li_nodes = ul.xpath("./li")
        amenities_count = len(li_nodes)

        # Beds
        bed_nodes = ul.xpath(
            './/li[contains(@class,"h-beds")]//span[contains(@class,"hz-figure")]/text()'
        )
        if bed_nodes:
            try:
                beds = int(bed_nodes[0])
            except ValueError:
                pass

        # Baths
        bath_nodes = ul.xpath(
            './/li[contains(@class,"h-baths")]//span[contains(@class,"hz-figure")]/text()'
        )
        if bath_nodes:
            try:
                bathrooms = int(bath_nodes[0])
            except ValueError:
                pass

    # In this theme, "Bed" is usually bedrooms, so we mirror it
    if beds is not None:
        bedrooms = beds

    # The site doesn't show rating, reviews, or minimum nights on the card,
    # so we default them.
    rating = None
    review_count = 0
    minimum_nights = 1

    listing_id = extract_listing_id_from_url(url) if url else None

    return {
        "platform": "LebanonRental",
        "listing_id": listing_id,
        "title": title,
        "city": city,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "beds": beds,
        "price": raw_price,
        "price_per_night_usd": price_per_night_usd,
        "rating": rating,
        "review_count": review_count,
        "amenities_count": amenities_count,
        "minimum_nights": minimum_nights,
        "url": url,
        "last_scraped": datetime.utcnow().date().isoformat(),
    }


def iter_listing_pages(start_url: str):
    """
    Iterate over all pages of /properties-for-rent/.
    This assumes pagination like:
      /properties-for-rent/
      /properties-for-rent/page/2/
      /properties-for-rent/page/3/
    If their pattern is different, just tweak the URL in here.
    """
    page = 1
    while True:
        if page == 1:
            url = start_url
        else:
            url = f"{start_url.rstrip('/')}/page/{page}/"

        try:
            tree = get_tree(url)
        except Exception as e:
            print(f"  -> stop: error fetching page {page}: {e}")
            break

        # Each big card starts with this class (from your HTML)
        cards = tree.xpath(
            '//div[contains(@class,"item-listing-wrap") and '
            'contains(@class,"item-wrap-v9")]'
        )
        if not cards:
            print("  -> no cards found, stopping pagination.")
            break

        yield cards
        page += 1
        time.sleep(1)


def scrape_lebanon_rental(output_csv: str = "lebanon_rental_raw.csv"):
    """
    Scrape all cards from /properties-for-rent/ into a CSV
    with the team’s columns.
    """
    fieldnames = [
        "platform",
        "listing_id",
        "title",
        "city",
        "bedrooms",
        "bathrooms",
        "beds",
        "price",
        "price_per_night_usd",
        "rating",
        "review_count",
        "amenities_count",
        "minimum_nights",
        "url",
        "last_scraped",
    ]

    total_rows = 0
    seen_ids = set()

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for cards in iter_listing_pages(BASE_URL):
            for card in cards:
                row = parse_card(card)
                # avoid duplicates if same listing appears twice
                key = row["listing_id"] or row["url"]
                if key in seen_ids:
                    continue
                seen_ids.add(key)

                writer.writerow(row)
                total_rows += 1

    print(f"\nDone. Wrote {total_rows} rows to {output_csv}")


if __name__ == "__main__":
    scrape_lebanon_rental()
