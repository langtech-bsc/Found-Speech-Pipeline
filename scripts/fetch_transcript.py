#!/usr/bin/env python3
"""Fetch a YouTube transcript and write it as a TSV line for the pipeline."""
import sys
from youtube_transcript_api import YouTubeTranscriptApi

VIDEO_ID = sys.argv[1] if len(sys.argv) > 1 else "PKuuatqwz00"
LANG     = sys.argv[2] if len(sys.argv) > 2 else "es"
WAV_PATH = f"/app/ingestion/{VIDEO_ID}.wav"

yta = YouTubeTranscriptApi()
transcript_list = yta.list(VIDEO_ID)
transcript = transcript_list.find_generated_transcript([LANG]).fetch()

# Each snippet is a FetchedTranscriptSnippet with a .text attribute
texts = [snippet.text for snippet in transcript]
full_text = " ".join(texts)

print(f"{WAV_PATH}\t{full_text}")
