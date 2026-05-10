from pathlib import Path
import urllib.request

DATA_URL = "https://github.com/glavrenov124/news-post-popularity-prediction/releases/download/data/posts_vk.csv"
OUTPUT_PATH = Path("data/raw/posts_vk.csv")


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        print(f"Dataset already exists: {OUTPUT_PATH}")
        return

    print("Downloading dataset...")
    urllib.request.urlretrieve(DATA_URL, OUTPUT_PATH)
    print(f"Dataset saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()