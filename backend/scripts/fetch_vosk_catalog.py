#!/usr/bin/env python3
"""Liest den offiziellen Vosk-Modellkatalog und schreibt app/vendor/vosk-catalog.json."""

import hashlib
import json
import re
import sys
import urllib.request
from datetime import date
from pathlib import Path
from urllib.parse import urljoin

SOURCE = "https://alphacephei.com/vosk/models"
OUT = Path(__file__).resolve().parent.parent / "app" / "vendor" / "vosk-catalog.json"

LANGUAGES = {
    "English": "Englisch", "Indian English": "Englisch (Indien)", "Chinese": "Chinesisch",
    "Russian": "Russisch", "French": "Französisch", "German": "Deutsch", "Spanish": "Spanisch",
    "Portuguese/Brazilian Portuguese": "Portugiesisch", "Greek": "Griechisch", "Turkish": "Türkisch",
    "Vietnamese": "Vietnamesisch", "Italian": "Italienisch", "Dutch": "Niederländisch",
    "Catalan": "Katalanisch", "Arabic": "Arabisch", "Arabic Tunisian": "Arabisch (Tunesien)",
    "Farsi": "Persisch", "Filipino": "Filipino", "Ukrainian": "Ukrainisch", "Kazakh": "Kasachisch",
    "Swedish": "Schwedisch", "Japanese": "Japanisch", "Esperanto": "Esperanto", "Hindi": "Hindi",
    "Czech": "Tschechisch", "Polish": "Polnisch", "Uzbek": "Usbekisch", "Korean": "Koreanisch",
    "Breton": "Bretonisch", "Gujarati": "Gujarati", "Tajik": "Tadschikisch", "Telugu": "Telugu",
    "Kyrgyz": "Kirgisisch", "Georgian": "Georgisch",
    "Speaker identification model": "Sprechererkennung",
}


def plain(value: str) -> str:
    value = re.sub(r"<[^>]*>", "", value).replace("&amp;", "&").replace("&nbsp;", " ")
    value = re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), value)
    return value.strip()


def main() -> int:
    with urllib.request.urlopen(SOURCE, timeout=60) as response:
        html = response.read().decode("utf-8", "replace")
    models = []
    tables = re.findall(r"<table\b[^>]*>(.*?)</table>", html, re.S)
    for index, table in enumerate(tables):
        group = ""
        for row in re.findall(r"<tr\b[^>]*>(.*?)</tr>", table, re.S):
            cells = re.findall(r"<td\b[^>]*>(.*?)</td>", row, re.S)
            if len(cells) < 3:
                continue
            link = re.search(r"<a\b[^>]*href=\"([^\"]+)\"[^>]*>(.*?)</a>", cells[0], re.S)
            if not link:
                heading = plain(cells[0])
                if heading:
                    group = heading
                continue
            model_id = plain(link.group(2))
            if not re.match(r"^vosk-(model|recasepunc)-", model_id):
                continue
            size = plain(cells[1])
            numeric = re.match(r"([\d.]+)\s*([MG])", size, re.I)
            bytes_ = None
            if numeric:
                factor = 10 ** 9 if numeric.group(2).upper() == "G" else 10 ** 6
                bytes_ = float(numeric.group(1)) * factor
            license_ = plain(cells[-1])
            language_key = group.removesuffix(" Other")
            kind = "punctuation" if index > 0 else ("speaker" if group == "Speaker identification model" else "speech")
            models.append({
                "id": model_id,
                "url": urljoin(SOURCE, link.group(1)),
                "language": LANGUAGES.get(language_key, language_key),
                "size": size,
                "bytes": bytes_,
                "license": license_,
                "kind": kind,
                "older": group.endswith(" Other"),
            })
    if len(models) < 60 or not any(m["id"] == "vosk-model-de-tuda-0.6-900k" for m in models):
        print("Unerwartete Katalogstruktur – Quelle geändert?", file=sys.stderr)
        return 1
    catalog = {
        "source": SOURCE,
        "checkedAt": date.today().isoformat(),
        "sourceSha256": hashlib.sha256(html.encode()).hexdigest(),
        "models": models,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n")
    speech = sum(1 for m in models if m["kind"] == "speech")
    print(json.dumps({"total": len(models), "speech": speech, "out": str(OUT)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
