"""Worker function for parallel language detection. Used by notebook 13."""

from langdetect import detect_langs, DetectorFactory

DetectorFactory.seed = 42


def detect_one(row_tuple):
    """Detect language for a single (title, artist) pair."""
    title, artist = row_tuple
    text = f'{title} {artist}'.strip()
    if not text or len(text) < 3:
        return (None, 0.0, len(text))
    try:
        top = detect_langs(text)[0]
        return (top.lang, round(top.prob, 4), len(text))
    except Exception:
        return (None, 0.0, len(text))
