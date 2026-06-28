from playwright.sync_api import sync_playwright
import csv
import sys

username = sys.argv[1].strip().lower()
post_limit = int(sys.argv[2])

PROFILE_URL = f"https://www.instagram.com/{username}/?hl=en"

BASE_XPATH = "/html/body/div[1]/div/div/div[2]/div/div/div[1]/div[1]/div[2]/section/main/div/div[1]/div/div[2]/div/div[2]/div/div[3]/div[{}]/div/div/div[2]/div/div[1]/div/div[2]/span"

COMMENTS_PER_POST = 10
comments = []

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)

    page = browser.new_page(
        viewport={"width": 1366, "height": 900}
    )

    print(f"Opening Instagram page: {PROFILE_URL}")
    page.goto(PROFILE_URL, timeout=60000)
    page.wait_for_timeout(5000)

    links = page.locator("a").all()
    post_urls = []

    for link in links:
        try:
            href = link.get_attribute("href")

            if href and ("/p/" in href or "/reel/" in href):
                post_url = (
                    "https://www.instagram.com" + href
                    if href.startswith("/")
                    else href
                )

                if post_url not in post_urls:
                    post_urls.append(post_url)

            if len(post_urls) >= post_limit:
                break

        except:
            pass

    for post_number, post_url in enumerate(post_urls, start=1):
        print(f"Opening post {post_number}: {post_url}")

        page.goto(post_url, timeout=60000)
        page.wait_for_timeout(7000)

        print(f"Extracting comments from post {post_number}")

        for i in range(1, COMMENTS_PER_POST + 1):
            xpath = BASE_XPATH.format(i)

            try:
                comment = page.locator(f"xpath={xpath}").inner_text(timeout=2000)
                comment = comment.strip()

                if comment and comment not in comments:
                    comments.append(comment)

            except:
                pass

    browser.close()

with open("comments.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["comment"])

    for comment in comments:
        writer.writerow([comment])

print(f"Saved {len(comments)} comments to comments.csv")