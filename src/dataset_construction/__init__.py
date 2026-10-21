__all__ = ["download_memes"]


def __getattr__(name):
    if name == "download_memes":
        from dataset_construction.scrapers.reddit_scraping import download_memes

        return download_memes
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
