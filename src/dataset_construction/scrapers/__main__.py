"""Entry point that runs the enabled scrapers based on the config file.

Run with::

    python -m dataset_construction.scrapers

Each scraper is toggled via the ``enabled`` flag in ``scrapers/config.yaml``.
"""

from dataset_construction.scrapers import (
    imgflip_scraping,
    kym_scraping,
    reddit_scraping,
)
from dataset_construction.scrapers.common import (
    DEFAULT_CONFIG_PATH,
    load_config,
)


def main(config_path=DEFAULT_CONFIG_PATH):
    config = load_config(config_path)

    ran_any = False

    if config.get("reddit", {}).get("enabled", False):
        print("\n########## Running Reddit scraper ##########")
        reddit_scraping.run_from_config(config_path)
        ran_any = True

    if config.get("imgflip", {}).get("enabled", False):
        print("\n########## Running imgflip scraper ##########")
        imgflip_scraping.run_from_config(config_path)
        ran_any = True

    if config.get("kym", {}).get("enabled", False):
        print("\n########## Running KYM scraper ##########")
        kym_scraping.run_from_config(config_path)
        ran_any = True

    if not ran_any:
        print("No scrapers enabled in config. Set 'enabled: true' to run one.")


if __name__ == "__main__":
    main()
