import time

import requests

from dataset_construction.scrapers.common import (
    DEFAULT_CONFIG_PATH,
    HEADERS,
    SCRAPED_DIR,
    load_config,
)

REDDIT_DIR = SCRAPED_DIR / "reddit"

IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg")


def download_memes(subreddit, total_needed=5):
    """Download images from a subreddit into ``scraped/reddit/<subreddit>/``."""
    full_path = REDDIT_DIR / subreddit
    full_path.mkdir(parents=True, exist_ok=True)

    # Count files already in the folder to resume progress
    existing_files = len([f for f in full_path.iterdir() if not f.name.startswith(".")])
    count = existing_files

    before = int(time.time())

    print(
        f"Starting download for r/{subreddit}. "
        f"Already have {existing_files} images. Goal: {total_needed}"
    )

    while count < total_needed:
        url = (
            "https://api.pullpush.io/reddit/search/submission/"
            f"?subreddit={subreddit}&limit=100&before={before}"
        )

        try:
            response = requests.get(url, headers=HEADERS, timeout=20).json()
            posts = response.get("data", [])

            if not posts:
                print("No more posts found in the archives.")
                break

            for post in posts:
                if count >= total_needed:
                    break

                img_url = post.get("url")

                # Check the URL exists and is an image before downloading.
                if (
                    img_url
                    and isinstance(img_url, str)
                    and img_url.split("?", 1)[0].lower().endswith(IMAGE_EXTENSIONS)
                ):
                    try:
                        # Handles URLs with extra params (e.g. "?width=...").
                        file_ext = img_url.split(".")[-1].split("?")[0]
                        file_name = f"{post['id']}.{file_ext}"
                        file_save_path = full_path / file_name

                        # Skip if the file already exists.
                        if file_save_path.exists():
                            continue

                        img_data = requests.get(
                            img_url, headers=HEADERS, timeout=15
                        ).content
                        file_save_path.write_bytes(img_data)

                        count += 1
                        if count % 20 == 0:
                            print(f"Progress: {count}/{total_needed} images saved.")

                    except Exception:
                        continue

            # Move back in time based on the last post in the batch.
            if posts:
                before = posts[-1]["created_utc"]

            time.sleep(1.5)  # Be kind to the API.

        except Exception as e:
            print(f"Batch Error: {e}. Retrying in 5 seconds...")
            time.sleep(5)
            continue


def run_from_config(config_path=DEFAULT_CONFIG_PATH):
    """Download memes for every subreddit listed in the config file."""
    config = load_config(config_path)
    reddit_config = config.get("reddit", {})
    subreddits = reddit_config.get("subreddits", [])

    if not subreddits:
        raise ValueError(
            f"No subreddits defined under 'reddit' in config: {config_path}"
        )

    for entry in subreddits:
        subreddit = entry.get("name")
        total_needed = entry.get("count")

        if not subreddit or total_needed is None:
            print(f"Skipping invalid config entry: {entry}")
            continue

        print(f"\n=== Scraping r/{subreddit} (target: {total_needed}) ===")
        download_memes(subreddit, total_needed=total_needed)


if __name__ == "__main__":
    run_from_config()