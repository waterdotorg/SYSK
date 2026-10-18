#!/usr/bin/env python3
"""
SYSK RSS Transcript Downloader
Downloads transcripts directly from RSS feed podcast:transcript URLs
No web scraping needed - just direct API calls!
"""

import requests
from bs4 import BeautifulSoup
import os
import re
import time
from datetime import datetime

class SYSKTranscriptDownloader:
    def __init__(self):
        self.output_dir = "transcripts"
        self.rss_url = "https://omnycontent.com/d/playlist/e73c998e-6e60-432f-8610-ae210140c5b1/A91018A4-EA4F-4130-BF55-AE270180C327/44710ECC-10BB-48D1-93C7-AE270180C33E/podcast.rss"
        self.failed_episodes = []
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def fetch_rss_feed(self):
        """Fetch and parse RSS feed"""
        print("Fetching RSS feed...")
        try:
            response = requests.get(self.rss_url, timeout=30)
            if response.status_code == 200:
                print(f"✓ RSS feed fetched successfully\n")
                soup = BeautifulSoup(response.content, 'xml')
                items = soup.find_all('item')
                print(f"✓ Found {len(items)} episodes in RSS feed\n")
                return items
            else:
                print(f"✗ Failed to fetch RSS feed: {response.status_code}")
                return None
        except Exception as e:
            print(f"✗ Error fetching RSS feed: {e}")
            return None
    
    def extract_episode_data(self, item):
        """Extract episode data from RSS item"""
        try:
            # Get title
            title_elem = item.find('title')
            title = title_elem.text.strip() if title_elem else "Unknown_Title"
            
            # Get pubDate and convert to YYYY-MM-DD
            pubdate_elem = item.find('pubDate')
            if pubdate_elem:
                # Parse: "Thu, 02 Oct 2025 11:37:12 +0000"
                pubdate_str = pubdate_elem.text.strip()
                dt = datetime.strptime(pubdate_str, '%a, %d %b %Y %H:%M:%S %z')
                date_formatted = dt.strftime('%Y-%m-%d')
            else:
                date_formatted = "Unknown_Date"
            
            # Get duration (in seconds, convert to minutes)
            duration_elem = item.find('itunes:duration')
            if duration_elem:
                duration_seconds = int(duration_elem.text.strip())
                duration_minutes = round(duration_seconds / 60)
            else:
                duration_minutes = 0
            
            # Get episode page URL (link to the episode on the website)
            episode_url = None
            link_elem = item.find('link')
            if link_elem:
                episode_url = link_elem.text.strip()
            
            # Get audio URL (the actual MP3 file)
            audio_url = None
            enclosure_elem = item.find('enclosure')
            if enclosure_elem:
                audio_url = enclosure_elem.get('url', '')
            
            # Get transcript URL (TextWithTimestamps format only)
            transcript_url = None
            transcript_elems = item.find_all('podcast:transcript')
            for trans_elem in transcript_elems:
                trans_type = trans_elem.get('type', '')
                trans_url = trans_elem.get('url', '')
                if 'text/plain' in trans_type and 'TextWithTimestamps' in trans_url:
                    transcript_url = trans_url
                    break
            
            return {
                'title': title,
                'date': date_formatted,
                'duration_minutes': duration_minutes,
                'episode_url': episode_url,
                'audio_url': audio_url,
                'transcript_url': transcript_url
            }
        except Exception as e:
            return None
    
    def sanitize_filename(self, text, max_length=80):
        """Clean text for use in filename"""
        # Remove special characters
        text = re.sub(r'[^\w\s-]', '', text)
        # Replace spaces with underscores
        text = re.sub(r'\s+', '_', text)
        # Remove multiple underscores
        text = re.sub(r'_+', '_', text)
        # Trim to max length
        text = text[:max_length]
        # Remove trailing underscores
        text = text.strip('_')
        return text
    
    def download_transcript(self, url, episode_title):
        """Download transcript from API URL"""
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                transcript = response.text
                lines = transcript.split('\n')
                
                # Find where iHeart intro appears (if at all)
                iheart_intro_line = -1
                iheart_speaker_num = None
                
                for i, line in enumerate(lines):
                    if "Welcome to Stuff You Should Know, a production of iHeartRadio" in line or \
                       "Welcome to Stuff you should know, a production of iHeartRadio" in line:
                        iheart_intro_line = i
                        # Extract the speaker number from this line
                        if line.startswith("Speaker "):
                            iheart_speaker_num = line.split(":", 1)[0]  # e.g., "Speaker 1"
                        break
                
                # Build speaker map
                speaker_map = {}
                
                if iheart_intro_line >= 0 and iheart_speaker_num:
                    # iHeart intro found
                    speaker_map[iheart_speaker_num] = "iHeart Intro"
                    
                    # Count lines per speaker AFTER the intro to find the two main speakers
                    speaker_line_counts = {}
                    for i in range(iheart_intro_line + 1, len(lines)):
                        line = lines[i]
                        if line.startswith("Speaker "):
                            speaker_num = line.split(":", 1)[0]
                            if speaker_num != iheart_speaker_num:  # Don't count intro speaker
                                speaker_line_counts[speaker_num] = speaker_line_counts.get(speaker_num, 0) + 1
                    
                    # Sort speakers by line count (descending) and take top 2
                    main_speakers = sorted(speaker_line_counts.items(), key=lambda x: x[1], reverse=True)[:2]
                    
                    # Map the two main speakers to Josh and Chuck (in order of first appearance)
                    main_speaker_nums = [speaker for speaker, count in main_speakers]
                    
                    # Sort by first appearance
                    first_appearances = {}
                    for i in range(iheart_intro_line + 1, len(lines)):
                        line = lines[i]
                        if line.startswith("Speaker "):
                            speaker_num = line.split(":", 1)[0]
                            if speaker_num in main_speaker_nums and speaker_num not in first_appearances:
                                first_appearances[speaker_num] = i
                    
                    sorted_main_speakers = sorted(main_speaker_nums, key=lambda x: first_appearances.get(x, 999))
                    
                    # Assign Josh and Chuck
                    if len(sorted_main_speakers) >= 1:
                        speaker_map[sorted_main_speakers[0]] = "Josh"
                    if len(sorted_main_speakers) >= 2:
                        speaker_map[sorted_main_speakers[1]] = "Chuck"
                
                else:
                    # No iHeart intro - count all speakers and map top 2 to Josh/Chuck
                    speaker_line_counts = {}
                    for line in lines:
                        if line.startswith("Speaker "):
                            speaker_num = line.split(":", 1)[0]
                            speaker_line_counts[speaker_num] = speaker_line_counts.get(speaker_num, 0) + 1
                    
                    # Sort by line count and take top 2
                    main_speakers = sorted(speaker_line_counts.items(), key=lambda x: x[1], reverse=True)[:2]
                    main_speaker_nums = [speaker for speaker, count in main_speakers]
                    
                    # Sort by first appearance
                    first_appearances = {}
                    for i, line in enumerate(lines):
                        if line.startswith("Speaker "):
                            speaker_num = line.split(":", 1)[0]
                            if speaker_num in main_speaker_nums and speaker_num not in first_appearances:
                                first_appearances[speaker_num] = i
                    
                    sorted_main_speakers = sorted(main_speaker_nums, key=lambda x: first_appearances.get(x, 999))
                    
                    # Assign Josh and Chuck
                    if len(sorted_main_speakers) >= 1:
                        speaker_map[sorted_main_speakers[0]] = "Josh"
                    if len(sorted_main_speakers) >= 2:
                        speaker_map[sorted_main_speakers[1]] = "Chuck"
                
                # Apply speaker replacements
                for speaker_num, assigned_name in speaker_map.items():
                    transcript = transcript.replace(f"{speaker_num}:", f"{assigned_name}:")
                
                return transcript
            else:
                return None
        except Exception as e:
            return None
    
    def download_episodes(self, num_episodes):
        """Main download workflow"""
        # Fetch RSS feed
        items = self.fetch_rss_feed()
        if not items:
            print("ERROR: Could not fetch RSS feed. Exiting.")
            return
        
        # Determine how many to process
        if num_episodes == 'all':
            episodes_to_process = items
        else:
            episodes_to_process = items[:num_episodes]
        
        total = len(episodes_to_process)
        successful = 0
        failed = 0
        skipped = 0
        selects = 0
        
        print(f"{'='*80}")
        print(f"Starting download of {total} episodes...")
        print(f"{'='*80}\n")
        
        for idx, item in enumerate(episodes_to_process, 1):
            print(f"{'='*80}")
            print(f"Progress: {idx}/{total}")
            
            # Extract episode data
            episode_data = self.extract_episode_data(item)
            
            if not episode_data:
                print(f"✗ Failed to extract episode data\n")
                failed += 1
                continue
            
            print(f"Title: {episode_data['title']}")
            print(f"Date: {episode_data['date']}")
            print(f"Duration: {episode_data['duration_minutes']} minutes")
            
            # Skip Selects episodes (they are repeats)
            if episode_data['title'].startswith("Selects"):
                print(f"⊘ Skipping Selects episode (repeat)\n")
                selects += 1
                continue
            
            # Check if transcript URL exists
            if not episode_data['transcript_url']:
                print(f"✗ No TextWithTimestamps transcript available\n")
                self.failed_episodes.append({
                    'title': episode_data['title'],
                    'reason': 'No transcript URL'
                })
                failed += 1
                continue
            
            # Create filename
            title_clean = self.sanitize_filename(episode_data['title'])
            filename = f"SYSK_{episode_data['date']}_{title_clean}_{episode_data['duration_minutes']}min.txt"
            filepath = os.path.join(self.output_dir, filename)
            
            # Check if already exists
            if os.path.exists(filepath):
                print(f"⊘ Already exists, skipping\n")
                skipped += 1
                continue
            
            # Download transcript
            print(f"Downloading transcript...")
            transcript = self.download_transcript(episode_data['transcript_url'], episode_data['title'])
            
            if transcript and len(transcript) > 100:
                # Save with metadata header
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"Episode: {episode_data['title']}\n")
                    f.write(f"Date: {episode_data['date']}\n")
                    f.write(f"Duration: {episode_data['duration_minutes']} minutes\n")
                    if episode_data.get('episode_url'):
                        f.write(f"Episode URL: {episode_data['episode_url']}\n")
                    if episode_data.get('audio_url'):
                        f.write(f"Audio URL: {episode_data['audio_url']}\n")
                    f.write(f"Transcript URL: {episode_data['transcript_url']}\n")
                    f.write("\n" + "="*80 + "\n\n")
                    f.write(transcript)
                
                print(f"✓ Saved ({len(transcript)} chars)\n")
                successful += 1
            else:
                print(f"✗ Transcript download failed or empty\n")
                self.failed_episodes.append({
                    'title': episode_data['title'],
                    'reason': 'Download failed'
                })
                failed += 1
            
            # Respectful delay
            time.sleep(2)
        
        # Summary
        print(f"\n{'='*80}")
        print(f"DOWNLOAD COMPLETE!")
        print(f"{'='*80}")
        print(f"  Total Processed: {total}")
        print(f"  ✓ Successful: {successful}")
        print(f"  ⊘ Skipped (already exists): {skipped}")
        print(f"  ⊘ Skipped (Selects repeats): {selects}")
        print(f"  ✗ Failed: {failed}")
        
        if self.failed_episodes:
            print(f"\nFailed Episodes:")
            for ep in self.failed_episodes[:10]:  # Show first 10
                print(f"  - {ep['title']} ({ep['reason']})")
            if len(self.failed_episodes) > 10:
                print(f"  ... and {len(self.failed_episodes) - 10} more")

    def list_missing(self, since=None):
        """Compare the RSS feed to the transcripts folder and report what's missing.
        Read-only: downloads nothing. Selects (repeat) episodes are ignored.
        since: optional 'YYYY-MM-DD'; only report episodes published on/after it."""
        items = self.fetch_rss_feed()
        if not items:
            print("ERROR: Could not fetch RSS feed. Exiting.")
            return None
        missing = []
        deepest = 0  # RSS is newest-first; how far back a --limit run must reach
        for pos, item in enumerate(items, 1):
            ep = self.extract_episode_data(item)
            if not ep or ep['title'].startswith("Selects"):
                continue
            if since and ep['date'] < since:
                continue
            filename = f"SYSK_{ep['date']}_{self.sanitize_filename(ep['title'])}_{ep['duration_minutes']}min.txt"
            if not os.path.exists(os.path.join(self.output_dir, filename)):
                missing.append((filename, bool(ep['transcript_url'])))
                deepest = pos
        print(f"{'='*80}")
        print(f"{len(missing)} episode(s) in the RSS feed are NOT in ./{self.output_dir}:")
        print(f"{'='*80}")
        for filename, has_tx in missing:
            flag = "" if has_tx else "   (no transcript in feed yet)"
            print(f"  {filename}{flag}")
        if missing:
            print(f"\nTo fetch them all, run:  python3 {os.path.basename(__file__)} --limit {deepest}")
        return missing


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download SYSK transcripts from the RSS feed.")
    parser.add_argument("--check", action="store_true",
                        help="List RSS episodes missing from ./transcripts, download nothing.")
    parser.add_argument("--since", metavar="YYYY-MM-DD",
                        help="With --check: only report episodes published on/after this date.")
    parser.add_argument("--limit", type=int,
                        help="Process the newest N RSS items (existing files are skipped).")
    parser.add_argument("--all", action="store_true", help="Process every RSS item.")
    parser.add_argument("--headless", action="store_true",
                        help="Accepted for backward compatibility; no effect (no browser is used).")
    args = parser.parse_args()

    print("\n" + "="*80)
    print("SYSK RSS Transcript Downloader")
    print("="*80 + "\n")

    downloader = SYSKTranscriptDownloader()

    if args.check:
        downloader.list_missing(since=args.since)
        return
    if args.all:
        downloader.download_episodes('all')
        return
    if args.limit:
        downloader.download_episodes(args.limit)
        return

    print("How many episodes do you want to download?")
    print("  - Enter a number (1-2721)")
    print("  - Enter 'all' to download all episodes")
    print("  - Enter 'test' to download first 5 episodes\n")

    choice = input("Your choice: ").strip().lower()

    if choice == 'test':
        num_episodes = 5
    elif choice == 'all':
        confirm = input("\nDownload all ~2,721 episodes? This will take several hours. (y/n): ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return
        num_episodes = 'all'
    else:
        try:
            num_episodes = int(choice)
            if num_episodes < 1 or num_episodes > 2721:
                print("Invalid number. Must be between 1 and 2721.")
                return
        except ValueError:
            print("Invalid input. Please enter a number or 'all'.")
            return

    print("\n")
    downloader.download_episodes(num_episodes)


if __name__ == "__main__":
    main()
