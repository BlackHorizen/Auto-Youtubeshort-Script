# -*- coding: utf-8 -*- # Add encoding declaration

import os
import sys
import time
import random
import logging
import re
import shutil
import platform
import subprocess
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict # Added Optional, Dict

# --- Attempt Library Imports ---
# (Keep Imports as they are)
try:
    logging.debug("Importing core libraries...")
    from moviepy.editor import (VideoFileClip, AudioFileClip, ImageClip,
                                CompositeVideoClip, CompositeAudioClip, vfx)
    from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
    import numpy as np
    logging.debug("Core libraries imported.")
    import whisper
    logging.debug("Whisper imported.")
    import torch
    logging.debug("PyTorch imported.")
    # --- YouTube Upload Imports ---
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request # Needed for refresh
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError
    logging.debug("Google API libraries imported.")
except ImportError as e:
    # Setup basic logger for early exit if imports fail
    _basic_logger = logging.getLogger()
    _basic_logger.setLevel("CRITICAL")
    _basic_handler = logging.StreamHandler(sys.stdout)
    _basic_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    _basic_handler.setFormatter(_basic_formatter)
    _basic_logger.addHandler(_basic_handler)
    _basic_logger.critical(f"Import Error: {e}. Please install required libraries.")
    _basic_logger.critical("Needed: moviepy Pillow numpy-stl openai-whisper torch")
    _basic_logger.critical("Needed for Upload: google-api-python-client google-auth-httplib2 google-auth-oauthlib")
    sys.exit(1) # Exit code 1 for general error


# --- Exit Codes ---
EXIT_CODE_SUCCESS = 0
EXIT_CODE_GENERAL_ERROR = 1
EXIT_CODE_NO_STORIES = 2
EXIT_CODE_PRE_RUN_FAILURE = 3
EXIT_CODE_UPLOAD_FAILURE = 4
EXIT_CODE_GENERATION_FAILURE = 5 # Specific for video gen error


# --- Configuration Class ---
# (Keep Config class as it is)
@dataclass
class Config:
    # ===============================================================
    # === Primary Adjustable Settings ===
    # ===============================================================
    # --- Source Text ---
    _TEXT_FILE_REL_PATH: str = os.path.join("input_data", "text", "Text.txt") # Relative input text file

    # --- Username Source ---
    _USERNAME_FILE_REL_PATH: str = os.path.join("input_data", "Usernames", "Usernames.txt") # Usernames file

    # --- TTS Settings (Coqui) ---
    COQUI_REFERENCE_AUDIO: str = r"D:\TTS Youtube\Coqui TTS\Voices\EllevenLabs\Tylor.mp3" # Voice clone sample
    AUDIO_SPEED_MULTIPLIER: float = 1.2 # Speed multiplier applied *after* TTS generation

    # --- Text Overlay Appearance ---
    FONT_MAIN_SIZE: int = 33
    FONT_HEADER_SIZE: int = 35
    AVATAR_ENABLED: bool = True   # Display the user avatar icon?
    AVATAR_SIZE: int = 60       # Pixel size (width & height) of the avatar if enabled
    HEADER_AVATAR_MARGIN: int = 10 # Space between avatar and username block
    HEADER_ELEMENT_SPACING: int = 8 # Spacing between Username and Time text

    # --- Music Settings ---
    MUSIC_ENABLED: bool = True
    MUSIC_FILE: Optional[str] = "UpBeat.mp3" # Filename in MUSIC_DIR
    MUSIC_VOLUME: float = 0.65

    # --- Video Timing & Output ---
    MAX_DURATION_SECONDS: float = 59.0
    OUTPUT_FPS: int = 30
    END_PADDING_SECONDS: float = 1.0

    # --- Background Video Source ---
    USE_LONG_BACKGROUND_VIDEO: bool = True
    LONG_BACKGROUND_VIDEO_PATH: str = r"D:\TTS Youtube\CraftShorts\input_data\Background\Long.mp4" # Absolute path ok here
    BACKGROUND_VID_DIR_SHORT: str = field(init=False) # Set in __post_init__

    # ===============================================================
    # === YouTube Upload Settings ===
    # ===============================================================
    ENABLE_YOUTUBE_UPLOAD: bool = True  # Set to False to disable automatic upload
    _YT_DATA_DIR_REL_PATH: str = "youtube_upload_data" # Relative dir for YT data files
    _YT_TITLE_FILE_REL_PATH: str = os.path.join(_YT_DATA_DIR_REL_PATH, "titles.txt")
    _YT_DESCRIPTION_FILE_REL_PATH: str = os.path.join(_YT_DATA_DIR_REL_PATH, "descriptions.txt")
    _YT_USED_PARTS_FILE_REL_PATH: str = os.path.join(_YT_DATA_DIR_REL_PATH, "used_parts.txt")
    _YT_CLIENT_SECRETS_REL_PATH: str = "client_secrets.json" # Expect in base dir
    _YT_TOKEN_REL_PATH: str = "token.json"                 # Expect in base dir
    # --- !!! MODIFIED TAGS !!! ---
    # Customize this list with tags you want. Removed #shorts and #AI.
    YT_VIDEO_TAGS: List[str] = field(default_factory=lambda: ["reddit", "askreddit", "storytime"]) # Example tags
    YT_VIDEO_CATEGORY_ID: str = "24" # 24=Entertainment, 22=People&Blogs, 27=Education, 28=SciTech
    YT_VIDEO_PRIVACY_STATUS: str = "public" # Options: "public", "private", "unlisted"
    YT_SELF_DECLARED_MADE_FOR_KIDS: bool = False
    YT_UPLOAD_SCOPES: List[str] = field(default_factory=lambda: ["https://www.googleapis.com/auth/youtube.upload"])


    # ===============================================================
    # === Paths & Directories (Derived) ===
    # ===============================================================
    BASE_DIR: str = field(default_factory=lambda: os.path.dirname(os.path.abspath(__file__)), init=False)
    INPUT_DIR: str = field(init=False)
    OUTPUT_DIR: str = field(init=False)
    USED_TEXT_DIR: str = field(init=False)
    TEMP_DIR_BASE: str = field(init=False)
    MUSIC_DIR: str = field(init=False)
    _AVATAR_REL_PATH: str = os.path.join("input_data", "Redditman", "Reddit.png") # Relative path for avatar
    AVATAR_PATH: Optional[str] = field(init=False) # Full path set in __post_init__
    USED_TEXT_FILE: str = field(init=False)
    TEXT_FILE_PATH: str = field(init=False)
    USERNAME_FILE_PATH: str = field(init=False)
    # YouTube Paths (Absolute)
    YT_DATA_DIR_PATH: str = field(init=False)
    YT_TITLE_FILE_PATH: str = field(init=False)
    YT_DESCRIPTION_FILE_PATH: str = field(init=False)
    YT_USED_PARTS_FILE_PATH: str = field(init=False)
    YT_CLIENT_SECRETS_FILE: str = field(init=False)
    YT_TOKEN_FILE: str = field(init=False)


    # ===============================================================
    # === External Tools & Advanced Settings ===
    # ===============================================================
    COQUI_SCRIPT_PATH: str = r"D:\TTS Youtube\Coqui TTS\Main.py"
    COQUI_LANG: str = "en"
    COQUI_ESPEAK_PATH: str = r"D:\TTS Youtube\Coqui TTS\espeak-ng-1.52.0"
    TTS_TIMEOUT_SECONDS: int = 480
    TTS_EXPECTED_SAMPLE_RATE: int = 24000
    WHISPER_MODEL: str = "medium"
    WHISPER_CONFIDENCE_THRESHOLD: float = 0.6
    OUTPUT_WIDTH: int = 1080
    OUTPUT_HEIGHT: int = 1920

    # --- Text Overlay Style (Less Frequent Changes) ---
    FONT_MAIN_NAME: str = "Arial"
    FONT_HEADER_NAME: str = "Arial Bold"
    FONT_FALLBACKS: List[str] = field(default_factory=lambda: ["Verdana", "DejaVu Sans"])
    FONT_MAIN_COLOR: Tuple[int, int, int, int] = (255, 255, 255, 255) # White RGBA
    FONT_HEADER_COLOR: Tuple[int, int, int, int] = (200, 200, 200, 255) # Light Grey RGBA
    TEXT_LINE_BG_COLOR: Tuple[int, int, int, int] = (50, 50, 50, 255) # Dark Grey RGBA (OPAQUE)
    TEXT_MARGIN_TOP: int = 180
    TEXT_MARGIN_SIDES: int = 50
    TEXT_BOX_PADDING_X: int = 15
    TEXT_BOX_PADDING_Y: int = 5
    LINE_SPACING: int = 0
    REVEAL_TIMING_ADJUST: float = 0.0
    HEADER_VERTICAL_PADDING: int = 0 # Space between header and main text blocks

    # --- Header Content ---
    HEADER_USERNAME_OPTIONS: List[str] = field(default_factory=lambda: [f"u/Redditor{random.randint(100,999)}" for _ in range(10)])
    HEADER_TIME_OPTIONS: List[str] = field(default_factory=lambda: [f"{random.randint(1, 12)}h ago"])

    # --- System & Debug ---
    LOG_LEVEL: str = "INFO"
    STANDARD_AUDIO_FPS: int = 44100

    # --- Path Initializer ---
    def __post_init__(self):
        """Construct absolute paths after BASE_DIR is initialized."""
        self.INPUT_DIR = os.path.join(self.BASE_DIR, "input_data")
        self.OUTPUT_DIR = os.path.join(self.BASE_DIR, "output")
        self.USED_TEXT_DIR = os.path.join(self.BASE_DIR, "used_text")
        self.TEMP_DIR_BASE = os.path.join(self.BASE_DIR, "Running")
        self.BACKGROUND_VID_DIR_SHORT = os.path.join(self.INPUT_DIR, "Background")
        self.MUSIC_DIR = os.path.join(self.INPUT_DIR, "Music")
        self.AVATAR_PATH = os.path.join(self.BASE_DIR, self._AVATAR_REL_PATH) if self.AVATAR_ENABLED else None
        self.USED_TEXT_FILE = os.path.join(self.USED_TEXT_DIR, "Used_Text.txt")
        self.TEXT_FILE_PATH = os.path.join(self.BASE_DIR, self._TEXT_FILE_REL_PATH)
        self.USERNAME_FILE_PATH = os.path.join(self.BASE_DIR, self._USERNAME_FILE_REL_PATH)

        # Ensure LONG_BACKGROUND_VIDEO_PATH is absolute if set and relative
        if self.USE_LONG_BACKGROUND_VIDEO and self.LONG_BACKGROUND_VIDEO_PATH and not os.path.isabs(self.LONG_BACKGROUND_VIDEO_PATH):
             self.LONG_BACKGROUND_VIDEO_PATH = os.path.abspath(os.path.join(self.BASE_DIR, self.LONG_BACKGROUND_VIDEO_PATH))

        # Construct YouTube absolute paths
        self.YT_DATA_DIR_PATH = os.path.join(self.BASE_DIR, self._YT_DATA_DIR_REL_PATH)
        self.YT_TITLE_FILE_PATH = os.path.join(self.BASE_DIR, self._YT_TITLE_FILE_REL_PATH)
        self.YT_DESCRIPTION_FILE_PATH = os.path.join(self.BASE_DIR, self._YT_DESCRIPTION_FILE_REL_PATH)
        self.YT_USED_PARTS_FILE_PATH = os.path.join(self.BASE_DIR, self._YT_USED_PARTS_FILE_REL_PATH)
        self.YT_CLIENT_SECRETS_FILE = os.path.join(self.BASE_DIR, self._YT_CLIENT_SECRETS_REL_PATH)
        self.YT_TOKEN_FILE = os.path.join(self.BASE_DIR, self._YT_TOKEN_REL_PATH)

# --- Global Variables ---
config = Config()
run_identifier = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
temp_dir = ""

# --- Logging Setup ---
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
# Ensure logger level is set early for import logging
logger.setLevel(config.LOG_LEVEL.upper())
# Remove existing handlers if any (useful for reruns in interactive sessions)
for handler in logger.handlers[:]: logger.removeHandler(handler)
# Setup stream handler (console output)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

# --- FFmpeg Check ---
try:
    logger.debug("Checking for FFmpeg..."); use_shell = platform.system() == 'Windows'
    subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True, timeout=10, shell=use_shell)
    logger.info("FFmpeg found in PATH.")
except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
    logger.critical(f"FATAL: FFmpeg not found or error running check ({e}). Please ensure FFmpeg is installed and in your system's PATH.")
    sys.exit(EXIT_CODE_PRE_RUN_FAILURE)


# --- Helper Functions ---
# (Keep all helper functions: safe_close, find_font, get_font, wrap_text, transcribe_with_whisper, select_random_file, process_background_video, calculate_line_reveal_times, generate_text_overlays, safe_create_dir, get_random_line, get_unique_part_number, authenticate_youtube, upload_video, cleanup_temp_dir, load_usernames_from_file)
# --- >>> HELPER FUNCTIONS DEFINED HERE <<< ---
def safe_close(clip):
    if clip and hasattr(clip, 'close') and callable(clip.close):
        try: clip.close(); logger.debug(f"Closed clip object: {repr(clip)}")
        except Exception as e: logger.debug(f"Error closing clip {repr(clip)}: {e}")

def find_font(font_name: str) -> Optional[str]:
    system = platform.system(); font_name_lower = font_name.lower()
    logger.debug(f"Searching for font: '{font_name}' on {system}")
    if os.path.isfile(font_name): return font_name # Direct path check first

    font_dirs = []
    # System-specific font directory search paths
    if system == "Windows":
        windir = os.environ.get('WINDIR', 'C:\\Windows'); font_dirs.append(os.path.join(windir, 'Fonts'))
        localappdata = os.environ.get('LOCALAPPDATA')
        if localappdata: font_dirs.append(os.path.join(localappdata, 'Microsoft', 'Windows', 'Fonts'))
    elif system == "Linux":
        font_dirs = ["/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts"), os.path.expanduser("~/.local/share/fonts")]
    elif system == "Darwin": # macOS
        font_dirs = ["/System/Library/Fonts", "/Library/Fonts", os.path.expanduser("~/Library/Fonts")]
    else:
        logger.warning(f"Unsupported OS '{system}' for automatic font searching.")

    logger.debug(f"Checking font directories: {font_dirs}")
    common_extensions = [".ttf", ".otf", ".ttc"]
    base_name_cleaned = re.sub(r'[\s_-]', '', font_name_lower)
    # Create variations (e.g., ArialBold -> arialbd, Arial Italic -> ariali) for broader matching
    variations = {base_name_cleaned, base_name_cleaned.replace('bold', 'bd'), base_name_cleaned.replace('italic', 'i'), base_name_cleaned.replace('regular', 'rg')}

    found_paths = []
    for directory in font_dirs:
        if not os.path.isdir(directory): continue
        try:
            # Walk through directory, including subdirs, follow symlinks
            for root, _, files in os.walk(directory, followlinks=True):
                for filename in files:
                    fname_lower = filename.lower()
                    base, ext = os.path.splitext(fname_lower)
                    if ext in common_extensions:
                        file_base_cleaned = re.sub(r'[\s-]', '', base)
                        # Check if cleaned filename matches any variation
                        for var_base in variations:
                            if file_base_cleaned == var_base or file_base_cleaned.startswith(var_base):
                                found_paths.append(os.path.join(root, filename))
                                break # Found a match for this file, move to next file
        except OSError as walk_err: logger.warning(f"Error walking font directory '{directory}': {walk_err}")
        except Exception as walk_e: logger.warning(f"Unexpected error walking font dir '{directory}': {walk_e}")

    if found_paths:
        # Prioritize exact base name matches (case-insensitive, ignoring spaces/hyphens)
        exact_matches = [p for p in found_paths if re.sub(r'[\s_-]', '', os.path.splitext(os.path.basename(p).lower())[0]) == base_name_cleaned]
        if exact_matches:
            logger.debug(f"Found exact font match for '{font_name}': {exact_matches[0]}")
            return exact_matches[0]
        else:
            # If no exact match, return the first potential match found
            logger.debug(f"Found potential font match for '{font_name}': {found_paths[0]}")
            return found_paths[0]

    # If not found in system paths, try fallback names recursively
    logger.warning(f"Font '{font_name}' not found in system directories. Trying fallbacks: {config.FONT_FALLBACKS}")
    checked_fallbacks = {font_name} # Keep track to avoid infinite recursion
    for fallback in config.FONT_FALLBACKS:
        if fallback in checked_fallbacks: continue
        checked_fallbacks.add(fallback)
        if os.path.isfile(fallback): # Check if fallback is a direct path
            logger.info(f"Using direct path fallback font: {fallback}")
            return fallback
        logger.debug(f"Searching for fallback font name: '{fallback}'")
        fallback_path = find_font(fallback) # Recursive call
        if fallback_path:
            logger.info(f"Found fallback font '{fallback}' at path: {fallback_path}")
            return fallback_path

    logger.error(f"Font '{font_name}' and all fallbacks could not be found.")
    return None

def get_font(font_name: str, size: int) -> Optional[ImageFont.FreeTypeFont]:
    font_path = find_font(font_name)
    if not font_path:
        logger.error(f"Could not find or load font path for '{font_name}'.")
        return None
    try:
        logger.debug(f"Attempting to load font: {font_path} size {size} (using RAQM layout engine)")
        # RAQM provides better complex script handling (Arabic, Indic, etc.)
        font_object = ImageFont.truetype(font_path, size, layout_engine=ImageFont.Layout.RAQM)
        logger.info(f"Successfully loaded font '{font_name}' using RAQM layout.")
        return font_object
    except ValueError as e_raqm:
        logger.warning(f"RAQM layout engine failed for '{font_path}': {e_raqm}. Trying basic layout.")
    except ImportError:
         logger.warning(f"RAQM layout engine not available (requires FriBiDi, HarfBuzz). Trying basic layout.")
    except Exception as e_initial:
        logger.warning(f"Initial font load attempt failed for '{font_path}': {e_initial}. Trying basic layout.")

    # Fallback to basic layout engine if RAQM fails or isn't available
    try:
        logger.debug(f"Attempting to load font: {font_path} size {size} (using basic layout engine)")
        font_object = ImageFont.truetype(font_path, size)
        logger.info(f"Successfully loaded font '{font_name}' using basic layout.")
        return font_object
    except Exception as e_basic:
        logger.error(f"Failed to load font '{font_path}' using basic layout engine: {e_basic}", exc_info=True)
        return None

def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    lines = []
    if not text or max_width <= 0:
        logger.warning("Wrap text called with empty text or invalid max_width.")
        return lines
    if not font:
        logger.error("Cannot wrap text: Invalid font object provided.")
        return text.split() # Basic fallback: split by whitespace

    logger.debug(f"Wrapping text (max_width={max_width}px): '{text[:60]}...'")
    try:
        # Create a dummy image and draw context for text measurement
        dummy_img = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(dummy_img)
    except Exception as e:
        logger.error(f"Failed to create PIL draw context for text wrapping: {e}")
        return text.split() # Fallback

    paragraphs = text.splitlines() # Preserve existing line breaks as paragraph breaks
    for i, paragraph in enumerate(paragraphs):
        paragraph = paragraph.strip() # Remove leading/trailing whitespace from paragraph
        if not paragraph:
            # Add an empty line if the previous paragraph wasn't empty (maintains spacing)
            if i > 0 and paragraphs[i-1].strip():
                lines.append("")
            continue # Skip processing empty paragraphs

        words = paragraph.split() # Split paragraph into words
        if not words: continue # Skip if paragraph only contained whitespace

        current_line = words[0]
        for word in words[1:]:
            test_line = f"{current_line} {word}"
            text_width = max_width + 1 # Default to overflow
            try:
                # Use textbbox for more accurate width, fallback to textlength
                # 'lt' anchor (left-top) is usually appropriate here
                bbox = draw.textbbox((0, 0), test_line, font=font, anchor="lt")
                text_width = bbox[2] - bbox[0] # width = right - left
            except AttributeError: # Older PIL might not have textbbox
                try: text_width = draw.textlength(test_line, font=font)
                except Exception: pass # Ignore errors in fallback measurement
            except Exception as e_measure:
                logger.warning(f"Error measuring text '{test_line[:30]}...': {e_measure}. Using fallback length.")
                try: text_width = draw.textlength(test_line, font=font) # Try length again
                except Exception: text_width = len(test_line) * font.size * 0.6 # Rough estimate

            # Handle cases where measurement might return 0 or negative
            if text_width <= 0:
                 try: text_width = draw.textlength(test_line, font=font)
                 except Exception: text_width = len(test_line) * font.size * 0.6 # Estimate again

            if text_width <= max_width:
                # Word fits on the current line
                current_line = test_line
            else:
                # Word does not fit, finalize the current line and start a new one
                lines.append(current_line)
                current_line = word
                # Check if the single word itself overflows (long word)
                single_word_width = max_width + 1
                try:
                    bbox_single = draw.textbbox((0,0), current_line, font=font, anchor="lt")
                    single_word_width = bbox_single[2] - bbox_single[0]
                except AttributeError:
                     try: single_word_width = draw.textlength(current_line, font=font)
                     except Exception: pass
                except Exception: pass # Ignore measurement errors
                if single_word_width <= 0: # Handle measurement errors for single word
                     try: single_word_width = draw.textlength(current_line, font=font)
                     except Exception: single_word_width = len(current_line) * font.size * 0.6
                if single_word_width > max_width:
                    logger.warning(f"Word exceeds max width: '{current_line}' ({single_word_width:.0f}px > {max_width}px). It will overflow.")
        # Append the last line of the paragraph
        lines.append(current_line)

    logger.debug(f"Text wrapping resulted in {len(lines)} lines.")
    return lines

def transcribe_with_whisper(audio_path: str) -> Optional[List[Dict]]:
    logger.info(f"Starting Whisper transcription for: {os.path.basename(audio_path)}")
    if not os.path.exists(audio_path):
        logger.error(f"Whisper error: Audio file not found at {audio_path}")
        return None
    try:
        # Determine device (GPU if available, else CPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Whisper model '{config.WHISPER_MODEL}' onto device '{device}'...")
        # Load the specified model size
        model = whisper.load_model(config.WHISPER_MODEL, device=device)
        logger.info("Whisper model loaded successfully.")

        start_time = time.time()
        logger.info("Starting audio transcription process...")
        # Perform transcription with word-level timestamps
        # fp16=True enables faster processing on CUDA GPUs, False for CPU/compatibility
        result = model.transcribe(audio_path, word_timestamps=True, fp16=(device == "cuda"))
        end_time = time.time()
        logger.info(f"Transcription complete. Elapsed time: {end_time - start_time:.2f} seconds.")

        # Extract word timestamps from the result structure
        word_timestamps = []
        if 'segments' in result:
            for segment in result.get('segments', []):
                 # Check if the segment itself contains word timestamps
                 if 'words' in segment:
                    for word_info in segment.get('words', []):
                        # Ensure essential keys exist and clean the word
                        word = word_info.get('word', '').strip()
                        start = word_info.get('start')
                        end = word_info.get('end')
                        # Whisper might use 'probability' or 'confidence' depending on version/settings
                        confidence = word_info.get('probability', word_info.get('confidence'))

                        if start is not None and end is not None and word: # Check if basic info is present
                            try:
                                # Convert timestamps and confidence to float, handle potential None for confidence
                                start, end = float(start), float(end)
                                conf_float = float(confidence) if confidence is not None else 1.0 # Assume 1.0 if missing
                            except (ValueError, TypeError) as e:
                                logger.warning(f"Skipping word '{word}' due to invalid timestamp/confidence data: {e}")
                                continue

                            # Filter based on confidence threshold
                            if conf_float >= config.WHISPER_CONFIDENCE_THRESHOLD:
                                word_timestamps.append({
                                    'word': word,
                                    'start': start,
                                    'end': end,
                                    'confidence': conf_float
                                })
                            # else: # Optional: Log words below threshold
                            #    logger.debug(f"Word excluded by confidence: '{word}' ({conf_float:.2f} < {config.WHISPER_CONFIDENCE_THRESHOLD})")

        if not word_timestamps:
            logger.warning("Whisper transcription finished but returned no valid word timestamps (or none met the confidence threshold).")
            return None # Indicate no usable timestamps found

        logger.info(f"Successfully extracted {len(word_timestamps)} word timestamps meeting the confidence threshold (>= {config.WHISPER_CONFIDENCE_THRESHOLD}).")
        return word_timestamps

    except Exception as e:
        logger.error(f"An error occurred during Whisper transcription: {e}", exc_info=True)
        return None

def select_random_file(directory: str, allowed_extensions: List[str] = ['.mp4', '.mov', '.avi', '.mkv', '.webm']) -> Optional[str]:
    logger.debug(f"Selecting random file with extensions {allowed_extensions} from directory: '{directory}'")
    if not os.path.isdir(directory):
        logger.error(f"Directory not found: {directory}")
        return None
    try:
        # Normalize extensions to lowercase for case-insensitive matching
        allowed_ext_lower = [ext.lower() for ext in allowed_extensions]
        # List files in the directory, filter by extension and ensure they are files (not directories)
        files = [f for f in os.listdir(directory)
                 if os.path.isfile(os.path.join(directory, f)) and
                    os.path.splitext(f)[1].lower() in allowed_ext_lower]

        if not files:
            logger.warning(f"No files with allowed extensions {allowed_extensions} found in directory: {directory}")
            return None

        # Choose a random file from the filtered list
        selected_file = random.choice(files)
        full_path = os.path.join(directory, selected_file)
        logger.info(f"Selected random file: {selected_file} (Full path: {full_path})")
        return full_path
    except OSError as e:
        logger.error(f"OS error while accessing directory '{directory}': {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error selecting random file from '{directory}': {e}")
        return None

def process_background_video(video_path: str, target_duration: float, is_long_video: bool = False) -> Optional[VideoFileClip]:
    logger.info(f"Processing background video: {os.path.basename(video_path)} (Target duration: {target_duration:.2f}s)")
    clip = None
    try:
        # Load video, disabling audio, setting target resolution early can help performance
        # Note: target_resolution might be ignored by some readers, resize/crop is more reliable later
        clip = VideoFileClip(video_path, audio=False, target_resolution=(config.OUTPUT_HEIGHT, config.OUTPUT_WIDTH))
        logger.debug(f"Initial background clip loaded. Size: {clip.size}, Duration: {clip.duration:.2f}s")

        # Basic validation
        if not clip.size or not all(clip.size) or clip.duration is None or clip.duration <= 0:
            raise ValueError(f"Loaded background video has invalid dimensions ({clip.size}) or duration ({clip.duration}).")

        original_duration = clip.duration

        # --- Duration Handling ---
        if is_long_video:
            if original_duration < target_duration:
                logger.warning(f"Long background video ({original_duration:.2f}s) is shorter than target duration ({target_duration:.2f}s). Using full video and looping if necessary (looping not implemented here, will be shorter).")
                # If looping is desired, it should be handled here or the clip duration set explicitly.
                # For now, we just use the available duration.
                target_duration = original_duration # Adjust target to what's available
            else:
                # Select a random segment from the long video
                max_start_time = original_duration - target_duration
                start_time = random.uniform(0, max_start_time)
                clip = clip.subclip(start_time, start_time + target_duration)
                logger.info(f"Extracted random segment from long video: {start_time:.2f}s - {start_time + target_duration:.2f}s")
        else: # Short background video
             if original_duration < target_duration:
                 logger.warning(f"Short background ({original_duration:.2f}s) shorter than target ({target_duration:.2f}s). Using full clip.")
                 target_duration = original_duration # Use the actual duration
             else:
                 # Use the beginning of the short clip
                 clip = clip.subclip(0, target_duration)
                 logger.info(f"Using first {target_duration:.2f}s of short background video.")


        # --- Aspect Ratio / Cropping / Resizing ---
        w, h = clip.size
        target_aspect = config.OUTPUT_WIDTH / config.OUTPUT_HEIGHT
        current_aspect = w / h

        if abs(current_aspect - target_aspect) > 0.01: # Check if aspect ratios differ significantly
            logger.debug(f"Background aspect ratio ({current_aspect:.2f}) differs from target ({target_aspect:.2f}). Cropping...")
            if current_aspect > target_aspect: # Video is wider than target (e.g., 16:9 source, 9:16 target)
                # Crop width, keep height
                new_width = int(round(h * target_aspect))
                x1 = max(0, int(round((w - new_width) / 2))) # Center crop horizontally
                x2 = min(w, x1 + new_width)
                if x1 < x2 and x2 <= w:
                    clip = clip.crop(x1=x1, y1=0, x2=x2, y2=h)
                    logger.debug(f"Cropped background width. New size: {clip.size}")
                else: logger.warning("Calculated invalid width crop parameters, skipping crop.")
            else: # Video is taller than target (e.g., 9:16 source, maybe slightly off, or square source)
                # Crop height, keep width
                new_height = int(round(w / target_aspect))
                y1 = max(0, int(round((h - new_height) / 2))) # Center crop vertically
                y2 = min(h, y1 + new_height)
                if y1 < y2 and y2 <= h:
                    clip = clip.crop(x1=0, y1=y1, x2=w, y2=y2)
                    logger.debug(f"Cropped background height. New size: {clip.size}")
                else: logger.warning("Calculated invalid height crop parameters, skipping crop.")

        # --- Final Resize & Duration Set ---
        target_size = (config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT)
        if list(clip.size) != list(target_size):
            logger.debug(f"Resizing background video from {clip.size} to {target_size}")
            clip = clip.resize(newsize=target_size)

        # Explicitly set final duration to avoid small floating point discrepancies
        clip = clip.set_duration(target_duration)

        # Final check on duration
        if abs(clip.duration - target_duration) > 0.05: # Allow tiny tolerance
            logger.warning(f"Background video final duration ({clip.duration:.3f}s) differs significantly from target ({target_duration:.3f}s) after processing!")

        logger.info(f"Background video processed successfully. Final duration: {clip.duration:.2f}s, Size: {clip.size}")
        return clip

    except Exception as e:
        logger.error(f"Error processing background video '{os.path.basename(video_path)}': {e}", exc_info=True)
        safe_close(clip) # Ensure clip is closed if an error occurred
        return None

def calculate_line_reveal_times(wrapped_lines: List[str], word_timestamps: List[Dict]) -> List[float]:
    line_reveals = [] # Stores the calculated start time for each line
    num_lines = len(wrapped_lines)
    num_timestamps = len(word_timestamps)
    logger.info(f"Calculating reveal times for {num_lines} text lines using {num_timestamps} word timestamps.")

    if not word_timestamps:
        logger.warning("No word timestamps provided for reveal time calculation. Setting all lines to reveal at time 0.0s.")
        return [0.0] * num_lines # Return a list of zeros if no timing data

    # Helper function to clean words for matching (lowercase, remove punctuation)
    def clean_word(w):
        if not isinstance(w, str): return ""
        # Remove leading/trailing punctuation, convert to lowercase
        cleaned = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', w)
        return cleaned.lower()

    current_ts_idx = 0      # Index of the timestamp we are currently looking at
    last_line_end_time = 0.0 # End time of the previous line, used as fallback start

    for line_idx, line in enumerate(wrapped_lines):
        words_on_line = line.split()

        # Handle empty lines: reveal them at the end time of the previous line
        if not words_on_line:
            reveal_time = last_line_end_time + config.REVEAL_TIMING_ADJUST
            line_reveals.append(max(0.0, reveal_time))
            logger.debug(f"Line {line_idx+1} (empty): Reveal time set to {line_reveals[-1]:.3f}s (based on previous line end)")
            continue

        # Find the start time of the first word on the line
        first_word_clean = clean_word(words_on_line[0])
        line_start_time = -1.0 # Initialize as not found

        # Search for the first word in the timestamps, starting from current_ts_idx
        # Limit search range to avoid excessive searching if a word is missed
        search_limit = min(current_ts_idx + 20, num_timestamps) # Look ahead 20 words max
        found_start_word = False
        for i in range(current_ts_idx, search_limit):
            ts_word_clean = clean_word(word_timestamps[i]['word'])
            if ts_word_clean == first_word_clean:
                line_start_time = word_timestamps[i]['start']
                # current_ts_idx = i # Update index tentatively (will be updated more reliably based on last word)
                found_start_word = True
                logger.debug(f"Line {line_idx+1}: First word '{words_on_line[0]}' matched timestamp index {i} at {line_start_time:.3f}s")
                break

        # If first word not found, use the end time of the previous line as start time
        if not found_start_word:
            line_start_time = last_line_end_time
            logger.warning(f"Line {line_idx+1}: First word '{words_on_line[0]}' not found in nearby timestamps (starting search from index {current_ts_idx}). Using previous line end time: {line_start_time:.3f}s")

        # Find the end time of the *last* word on the line for accurate line duration
        line_end_time = line_start_time # Default end time is the start time
        last_word_idx_on_line = -1     # Index of the timestamp corresponding to the last matched word on the line
        temp_search_idx = current_ts_idx # Use a temporary index for searching words within the line
        word_match_indices_on_line = []

        # Iterate through words on the current line to find their timestamps
        for word_on_line in words_on_line:
             word_clean = clean_word(word_on_line)
             if not word_clean: continue # Skip empty strings resulting from cleaning

             found_match_for_word = False
             # Search for the current word starting from temp_search_idx
             search_end_limit = min(temp_search_idx + 15, num_timestamps) # Look ahead reasonably
             for i in range(temp_search_idx, search_end_limit):
                 ts_word_clean = clean_word(word_timestamps[i]['word'])
                 if ts_word_clean == word_clean:
                     word_match_indices_on_line.append(i) # Store the index of the matched timestamp
                     temp_search_idx = i + 1 # Advance search index for the next word
                     found_match_for_word = True
                     # logger.debug(f"  - Word '{word_on_line}' matched ts index {i}") # Optional detailed log
                     break # Found the word, move to the next word on the line
             # if not found_match_for_word: # Optional: Log if a word wasn't found
             #      logger.debug(f"  - Word '{word_on_line}' not found near index {temp_search_idx}")

        # Determine the end time based on the *last* successfully matched word on the line
        if word_match_indices_on_line:
            last_word_idx_on_line = max(word_match_indices_on_line) # Get the highest index found for this line
            line_end_time = word_timestamps[last_word_idx_on_line]['end']
            # Crucially, update the main timestamp index to search from after the last matched word
            current_ts_idx = last_word_idx_on_line + 1
            logger.debug(f"Line {line_idx+1}: Last matched word '{word_timestamps[last_word_idx_on_line]['word']}' (ts index {last_word_idx_on_line}) ends at {line_end_time:.3f}s. Next search starts at {current_ts_idx}.")
        else:
            # If *no* words on the line were matched (unlikely if first word search works, but possible)
            # Advance the main index roughly based on the number of words to avoid getting stuck
            logger.warning(f"Line {line_idx+1}: No words on this line were matched in timestamps. End time defaults to start time {line_end_time:.3f}s. Advancing search index heuristically.")
            current_ts_idx = min(current_ts_idx + len(words_on_line), num_timestamps) # Move index forward

        # Calculate final reveal time for the line (start time + adjustment)
        final_reveal_time = max(0.0, line_start_time + config.REVEAL_TIMING_ADJUST)
        line_reveals.append(final_reveal_time)
        last_line_end_time = line_end_time # Update the end time for the next iteration

    # Final check: Ensure the number of reveal times matches the number of lines
    if len(line_reveals) != num_lines:
        logger.error(f"Mismatch between number of lines ({num_lines}) and calculated reveal times ({len(line_reveals)}). Padding with last time.")
        # Pad with the last calculated time if needed
        last_time = line_reveals[-1] if line_reveals else 0.0
        line_reveals.extend([last_time] * (num_lines - len(line_reveals)))

    logger.info(f"Calculated reveal times: {[(i+1, t) for i, t in enumerate(line_reveals)]}") # Log line number and time
    return line_reveals

# --- REVISED generate_text_overlays (Ensures RGB output for compatibility) ---
def generate_text_overlays(text_lines: List[str], line_start_times: List[float],
                           font_main: ImageFont.FreeTypeFont, font_header: ImageFont.FreeTypeFont,
                           total_duration: float) -> Optional[CompositeVideoClip]:
    """
    Generates text overlay clips with opaque backgrounds, ensuring final images
    are converted to RGB before creating ImageClips for broader compatibility
    with video codecs/players. Header and main text backgrounds are distinct but coordinated.
    """
    logger.info(f"Generating dynamic text overlays (RGB output mode) for header and {len(text_lines)} main text lines.")
    if not all([font_main, font_header]):
        logger.error("Overlay generation failed: Invalid Main or Header Font object provided.")
        return None
    if text_lines and len(text_lines) != len(line_start_times):
        mismatch_len = min(len(text_lines), len(line_start_times))
        logger.error(f"Mismatched main text lines ({len(text_lines)}) and start times ({len(line_start_times)}). Truncating to {mismatch_len} lines.")
        text_lines = text_lines[:mismatch_len]
        line_start_times = line_start_times[:mismatch_len]

    all_clips = [] # List to hold individual ImageClips for header and each line
    current_y = config.TEXT_MARGIN_TOP # Initial Y position for the top of the header block

    try:
        # Dummy image/draw context for measurements
        dummy_img = Image.new('RGBA', (1, 1))
        draw = ImageDraw.Draw(dummy_img)
    except Exception as e:
        logger.error(f"Failed to create PIL draw context for overlay measurements: {e}")
        return None

    # --- Measure Main Text First to Determine Max Width ---
    max_main_text_width = 0
    main_text_line_heights = [] # Store measured height of each line's text content
    if text_lines:
        logger.debug("Measuring main text lines for width and height...")
        for i, line in enumerate(text_lines):
            if not line.strip():
                # Assign a minimal height for empty lines to maintain some spacing if needed
                line_h = int(font_main.size * 0.5)
                line_w = 0
            else:
                try:
                    bbox = draw.textbbox((0,0), line, font=font_main, anchor="lt")
                    line_w = max(1, bbox[2] - bbox[0]) # Ensure width is at least 1
                    line_h = max(1, bbox[3] - bbox[1]) # Ensure height is at least 1
                    # Fallback height if measurement seems too small
                    if line_h <= 1: line_h = int(font_main.size * 1.2)
                except Exception as e_measure:
                    logger.warning(f"Error measuring main text line {i+1}: '{line[:30]}...'. Estimating dimensions. Error: {e_measure}")
                    line_w = len(line) * font_main.size * 0.6 # Rough estimate
                    line_h = int(font_main.size * 1.2) # Estimate height based on font size
            max_main_text_width = max(max_main_text_width, line_w)
            main_text_line_heights.append(line_h) # Store the text content height
        logger.debug(f"Maximum measured main text content width: {max_main_text_width:.0f}px")

    # Calculate the required background width (for both header and main text)
    # Based on the widest main text line + padding
    unified_bg_width = max(1, int(max_main_text_width + 2 * config.TEXT_BOX_PADDING_X))
    logger.debug(f"Calculated unified background width (including padding): {unified_bg_width}px")

    # --- Header Generation ---
    header_clip = None
    header_max_content_h = 0 # Track the tallest element within the header (text or avatar)
    avatar_pil = None          # PIL Image object for the avatar

    # 1. Prepare Header Content (Username, Time)
    try:
        username_display = random.choice(config.HEADER_USERNAME_OPTIONS)
        time_ago = random.choice(config.HEADER_TIME_OPTIONS)
        time_display = f"· {time_ago}" # Add dot separator
    except IndexError: # Handle empty options lists
         username_display, time_ago = "u/DefaultUser", "recently"
         time_display = f"· {time_ago}"
         logger.warning("Header username/time options list empty, using defaults.")
    except Exception as e:
        username_display, time_ago = "u/ErrorUser", "error"
        time_display = f"· {time_ago}"
        logger.error(f"Unexpected error getting header content: {e}")

    # 2. Measure Header Elements & Load Avatar
    try:
        # Measure Username
        username_bbox = draw.textbbox((0, 0), username_display, font=font_header, anchor="lt")
        username_height = max(1, username_bbox[3] - username_bbox[1])
        username_width_actual = max(1, username_bbox[2] - username_bbox[0]) # Get actual width
        header_max_content_h = max(header_max_content_h, username_height)
        logger.debug(f"Measured username '{username_display}': W={username_width_actual}, H={username_height}")

        # Measure Time
        time_bbox = draw.textbbox((0, 0), time_display, font=font_header, anchor="lt")
        time_height = max(1, time_bbox[3] - time_bbox[1])
        time_width_actual = max(1, time_bbox[2] - time_bbox[0]) # Get actual width
        header_max_content_h = max(header_max_content_h, time_height)
        logger.debug(f"Measured time '{time_display}': W={time_width_actual}, H={time_height}")

        # Load and Prepare Avatar (if enabled)
        if config.AVATAR_ENABLED and config.AVATAR_PATH and os.path.exists(config.AVATAR_PATH):
            try:
                with Image.open(config.AVATAR_PATH) as img:
                    # Convert to RGBA for transparency handling, resize with high quality downsampling
                    avatar_pil = img.convert("RGBA").resize((config.AVATAR_SIZE, config.AVATAR_SIZE), Image.Resampling.LANCZOS)
                header_max_content_h = max(header_max_content_h, config.AVATAR_SIZE) # Update max height if avatar is taller
                logger.debug(f"Loaded and resized avatar. Size: {avatar_pil.size}")
            except UnidentifiedImageError:
                 logger.warning(f"Could not identify image file format for avatar: {config.AVATAR_PATH}. Disabling avatar.")
                 avatar_pil = None
            except Exception as e:
                logger.warning(f"Failed to load or resize avatar from '{config.AVATAR_PATH}': {e}. Disabling avatar.")
                avatar_pil = None
        else:
             if config.AVATAR_ENABLED: logger.debug("Avatar disabled or path not found.")
             avatar_pil = None # Ensure it's None if disabled or missing

        # Fallback height if all measurements failed
        if header_max_content_h <= 1: header_max_content_h = int(font_header.size * 1.2)

        # Calculate total header height needed (content + vertical padding)
        header_total_height = int(header_max_content_h + 2 * config.TEXT_BOX_PADDING_Y)
        header_bg_width = unified_bg_width # Use the width determined from main text

        logger.debug(f"Header dimensions calculated: BG_Width={header_bg_width}, BG_Height={header_total_height}, Max_Content_Height={header_max_content_h}")

    except Exception as e:
        logger.error(f"Error measuring header elements or loading avatar: {e}. Header creation might fail.", exc_info=True)
        header_total_height = 0 # Indicate failure to create header


    # 3. Create Single Header Image Clip (if height > 0)
    if header_total_height > 0:
        try:
            # Create RGBA image first to handle potential avatar transparency
            header_img_rgba = Image.new('RGBA', (header_bg_width, header_total_height), (0, 0, 0, 0)) # Transparent base
            draw_h = ImageDraw.Draw(header_img_rgba)

            # Draw the opaque background rectangle
            bg_color_rgba = config.TEXT_LINE_BG_COLOR # Assumed to be RGBA tuple
            draw_h.rectangle(
                (0, 0, header_bg_width - 1, header_total_height - 1),
                fill=bg_color_rgba
            )

            # --- Position and Draw Header Elements ---
            current_x_on_header = config.TEXT_BOX_PADDING_X # Starting X position inside the header box

            # Paste Avatar (if loaded)
            if avatar_pil:
                # Calculate Y position to vertically center avatar within the padded content area
                avatar_y_on_header = config.TEXT_BOX_PADDING_Y + max(0, (header_max_content_h - config.AVATAR_SIZE) // 2)
                # Paste using RGBA mask for proper transparency handling
                header_img_rgba.paste(avatar_pil, (int(current_x_on_header), int(avatar_y_on_header)), avatar_pil)
                logger.debug(f"Pasted Avatar at ({int(current_x_on_header)}, {int(avatar_y_on_header)}) on header image.")
                current_x_on_header += config.AVATAR_SIZE + config.HEADER_AVATAR_MARGIN # Advance X position

            # Draw Username
            # Calculate Y position to vertically center username
            username_y_on_header = config.TEXT_BOX_PADDING_Y + max(0, (header_max_content_h - username_height) // 2)
            draw_h.text(
                (current_x_on_header, username_y_on_header),
                username_display,
                font=font_header,
                fill=config.FONT_HEADER_COLOR,
                anchor="lt" # Left-top anchor
            )
            logger.debug(f"Drew Username at ({current_x_on_header}, {username_y_on_header}) on header image.")
            current_x_on_header += username_width_actual + config.HEADER_ELEMENT_SPACING # Advance X using measured width

            # Draw Time
            # Calculate Y position to vertically center time text
            time_y_on_header = config.TEXT_BOX_PADDING_Y + max(0, (header_max_content_h - time_height) // 2)
            draw_h.text(
                (current_x_on_header, time_y_on_header),
                time_display,
                font=font_header,
                fill=config.FONT_HEADER_COLOR,
                anchor="lt" # Left-top anchor
            )
            logger.debug(f"Drew Time at ({current_x_on_header}, {time_y_on_header}) on header image.")

            # --- Convert Final Header Image to RGB ---
            # This step is crucial for compatibility as many video codecs expect RGB frames.
            # It effectively flattens the image onto a black background if there were transparent areas,
            # but since we drew an opaque bg_color_rgba, it just converts the format.
            header_img_rgb = header_img_rgba.convert('RGB')
            header_np = np.array(header_img_rgb) # Convert PIL Image to NumPy array for MoviePy

            # Validate numpy array shape (should be Height x Width x 3 channels)
            if header_np.ndim != 3 or header_np.shape[2] != 3:
                raise ValueError(f"Header image conversion to RGB numpy array failed. Shape: {header_np.shape}")

            # Create the MoviePy ImageClip from the RGB NumPy array
            header_clip = ImageClip(header_np, ismask=False, transparent=False) # Not a mask, not transparent
            header_clip = header_clip.set_position((config.TEXT_MARGIN_SIDES, current_y)) # Position on screen
            header_clip = header_clip.set_duration(total_duration) # Make it last the whole video
            header_clip = header_clip.set_start(0)                  # Start at time 0
            all_clips.append(header_clip)
            logger.info(f"Created single RGB header clip (Size: {header_bg_width}x{header_total_height}) positioned at y={current_y}")

            # Advance Y position for the main text block
            # Add the calculated total height of the header block
            current_y += header_total_height
            # Add the configured vertical padding *between* header and main text (if any)
            current_y += config.HEADER_VERTICAL_PADDING

        except Exception as e:
            logger.error(f"Failed to create header overlay clip: {e}", exc_info=True)
            # Advance Y position even if header fails, using estimated/calculated height,
            # to potentially avoid overlap with main text if it proceeds.
            current_y += header_total_height + config.HEADER_VERTICAL_PADDING


    # --- Main Text Line Processing (Convert lines to RGB) ---
    text_block_start_x = config.TEXT_MARGIN_SIDES # X position for all main text lines
    main_text_bg_width = unified_bg_width         # Use the unified width calculated earlier

    if text_lines:
        logger.debug(f"Generating {len(text_lines)} main text line overlay clips...")
        for i, line in enumerate(text_lines):
            start_time = max(0.0, line_start_times[i]) # Ensure start time is not negative
            start_time = min(start_time, total_duration - 0.01) # Ensure start time is within video duration bounds
            line_h_content = main_text_line_heights[i] # Get the stored text content height
            # Calculate total height for this line's background (content + vertical padding)
            line_h_total = int(line_h_content + 2 * config.TEXT_BOX_PADDING_Y)
            # Calculate the duration this line should be visible (from its start time to the end)
            clip_duration = max(0.01, total_duration - start_time) # Ensure minimum duration
            line_y = current_y # Y position for the top of this line's background box

            # Handle empty lines: just advance Y position for spacing, don't create a clip
            if not line.strip():
                current_y += line_h_content # Advance by the estimated empty line height
                logger.debug(f"Skipping clip generation for empty line {i+1}, advancing Y position.")
                continue

            try:
                # Create RGBA image for the line (allows transparent base if needed, though bg is opaque)
                line_img_rgba = Image.new('RGBA', (main_text_bg_width, line_h_total), (0, 0, 0, 0))
                draw_l = ImageDraw.Draw(line_img_rgba)

                # Draw the background rectangle for this line
                draw_l.rectangle(
                    (0, 0, main_text_bg_width - 1, line_h_total - 1),
                    fill=config.TEXT_LINE_BG_COLOR
                )

                # Calculate text position inside the background box (padded)
                text_x_pos = config.TEXT_BOX_PADDING_X
                text_y_pos = config.TEXT_BOX_PADDING_Y

                # Draw the text onto the background
                draw_l.text(
                    (text_x_pos, text_y_pos),
                    line,
                    font=font_main,
                    fill=config.FONT_MAIN_COLOR,
                    anchor="lt" # Left-top anchor
                )

                # --- Convert Line Image to RGB ---
                line_img_rgb = line_img_rgba.convert('RGB')
                line_np = np.array(line_img_rgb) # Convert to NumPy array

                # Validate numpy array shape
                if line_np.ndim != 3 or line_np.shape[2] != 3:
                    raise ValueError(f"Main text line {i+1} conversion to RGB numpy array failed. Shape: {line_np.shape}")

                # Create the MoviePy ImageClip from the RGB NumPy array
                line_clip = ImageClip(line_np, ismask=False, transparent=False)
                line_clip = line_clip.set_position((text_block_start_x, line_y)) # Set position
                line_clip = line_clip.set_start(start_time)                       # Set reveal time
                line_clip = line_clip.set_duration(clip_duration)                 # Set duration
                all_clips.append(line_clip)

                # Advance Y position for the next line (total height of current line + spacing)
                current_y += line_h_total + config.LINE_SPACING

            except Exception as e:
                logger.error(f"Failed to create overlay clip for main text line {i+1}: '{line[:30]}...'. Error: {e}", exc_info=True)
                # Attempt to advance Y position even on error to prevent stacking
                current_y += line_h_total + config.LINE_SPACING

    # --- Final Composite Video Clip ---
    if not all_clips:
        logger.error("No overlay clips (header or text lines) were generated. Cannot create composite.")
        return None

    try:
        logger.info(f"Compositing {len(all_clips)} overlay clips (header + text lines) onto a transparent background...")
        # Create a composite clip using the list of individual ImageClips
        # `size` specifies the final canvas size; `use_bgclip=False` implies a transparent background
        final_composite = CompositeVideoClip(
            all_clips,
            size=(config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT),
            use_bgclip=False
        ).set_duration(total_duration) # Ensure composite has the correct total duration

        logger.info("Overlay composite clip created successfully.")
        return final_composite
    except Exception as e:
        logger.error(f"Failed to composite the final overlay video clip: {e}", exc_info=True)
        # Attempt to clean up individual clips if compositing fails
        for clip in all_clips: safe_close(clip)
        return None

# --- YouTube Upload Helper Functions ---

def safe_create_dir(dir_path: str):
    """Creates a directory if it doesn't exist, handles potential errors."""
    if not dir_path:
        # Log or raise error? For now, log and return False
        logger.error("safe_create_dir received an empty directory path.")
        return False
    try:
        os.makedirs(dir_path, exist_ok=True)
        logger.debug(f"Ensured directory exists: {dir_path}")
        return True
    except OSError as e:
        logger.error(f"Failed to create directory '{dir_path}': {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error creating directory '{dir_path}': {e}")
        return False

def get_random_line(file_path: str) -> Optional[str]:
    """Reads a file and returns a random non-empty line."""
    global logger, config # Ensure access to global logger and config
    logger.debug(f"Attempting to read random line from: {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"File not found for get_random_line: {file_path}")
        # Try to create the parent directory (using config path for consistency)
        parent_dir = os.path.dirname(file_path)
        if not safe_create_dir(parent_dir):
             logger.warning(f"Could not create parent directory for {file_path}")
        # Create an empty file so it exists for next time
        try:
             with open(file_path, 'a', encoding='utf-8') as f: pass
             logger.info(f"Created empty file: {file_path}")
        except Exception as e_create:
             logger.warning(f"Could not create empty file {file_path}: {e_create}")
        return None # Return None if file initially doesn't exist

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            logger.warning(f"File is empty or contains only whitespace: {file_path}")
            return None
        chosen_line = random.choice(lines)
        logger.debug(f"Selected random line: '{chosen_line[:50]}...'")
        return chosen_line
    except Exception as e:
        logger.error(f"Error reading random line from {file_path}: {e}")
        return None

def get_unique_part_number() -> int:
    """Gets a random 3-digit number, ensuring it hasn't been used before."""
    global logger, config # Ensure access to global logger and config
    logger.debug(f"Getting unique part number using file: {config.YT_USED_PARTS_FILE_PATH}")
    used_parts = set()
    parts_file = config.YT_USED_PARTS_FILE_PATH
    parts_file_dir = os.path.dirname(parts_file)

    # Ensure directory exists before trying to read/write
    if not safe_create_dir(parts_file_dir):
        logger.error("Cannot proceed without directory for used parts file.")
        return 0 # Indicate failure

    try:
        if os.path.exists(parts_file):
            with open(parts_file, "r", encoding="utf-8") as f:
                used_parts = set(line.strip() for line in f if line.strip()) # Read and strip lines
            logger.debug(f"Loaded {len(used_parts)} used part numbers.")
        else:
             # Create the file if it doesn't exist
             with open(parts_file, 'a', encoding='utf-8') as f: pass
             logger.info(f"Created empty used parts file: {parts_file}")

    except Exception as e:
        logger.warning(f"Could not read used parts file '{parts_file}': {e}. Starting fresh for this run.")
        used_parts = set()

    attempts = 0
    max_attempts = 5000 # Prevent infinite loop
    while attempts < max_attempts:
        part = random.randint(100, 999)
        if str(part) not in used_parts:
            try:
                with open(parts_file, "a", encoding="utf-8") as f:
                    f.write(f"{part}\n")
                logger.info(f"Generated and recorded unique part number: {part}")
                return part
            except Exception as e:
                logger.error(f"Error writing new part number {part} to {parts_file}: {e}")
                # Don't return the part number if writing failed, try again
        attempts += 1

    logger.error(f"Failed to generate a unique part number after {max_attempts} attempts. Check {parts_file}.")
    return 0 # Fallback / Indicate failure

def authenticate_youtube() -> Optional[object]:
    """Handles YouTube API authentication using OAuth 2.0."""
    global logger, config # Ensure access to global logger and config
    logger.info("Authenticating with YouTube API...")
    creds = None
    token_file = config.YT_TOKEN_FILE
    secrets_file = config.YT_CLIENT_SECRETS_FILE
    scopes = config.YT_UPLOAD_SCOPES

    if os.path.exists(token_file):
        try:
            creds = Credentials.from_authorized_user_file(token_file, scopes)
            logger.debug(f"Loaded credentials from {token_file}")
        except Exception as e:
            logger.warning(f"Error loading credentials from {token_file}: {e}. Will attempt re-authentication.")
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.info("Credentials expired, attempting refresh...")
            try:
                creds.refresh(Request()) # Needs: from google.auth.transport.requests import Request
                logger.info("Credentials refreshed successfully.")
            except Exception as e:
                logger.warning(f"Failed to refresh credentials: {e}. Proceeding to request new authorization.")
                creds = None
        else:
            logger.info("No valid credentials found or refresh failed. Starting OAuth flow...")
            if not os.path.exists(secrets_file):
                logger.critical(f"FATAL: YouTube client secrets file not found at: {secrets_file}")
                logger.critical("Download it from Google Cloud Console and place it in the script's directory.")
                return None
            try:
                flow = InstalledAppFlow.from_client_secrets_file(secrets_file, scopes)
                creds = flow.run_local_server(port=0)
                logger.info("OAuth flow completed successfully.")
            except Exception as e:
                logger.critical(f"OAuth flow failed: {e}", exc_info=True)
                return None

        if creds:
            try:
                with open(token_file, "w") as token:
                    token.write(creds.to_json())
                logger.info(f"Credentials saved to {token_file}")
            except Exception as e:
                logger.error(f"Error saving credentials to {token_file}: {e}")

    if creds and creds.valid:
        logger.info("YouTube authentication successful.")
        try:
            youtube_service = build("youtube", "v3", credentials=creds)
            logger.info("YouTube API service object created.")
            return youtube_service
        except Exception as e:
            logger.error(f"Failed to build YouTube API service: {e}", exc_info=True)
            return None
    else:
        logger.error("Could not obtain valid YouTube credentials.")
        return None

def upload_video(youtube: object, video_path: str, title: str, description: str) -> Optional[str]:
    """Uploads the specified video file to YouTube."""
    global logger, config # Ensure access to global logger and config
    logger.info(f"Preparing to upload video: {os.path.basename(video_path)}")
    if not os.path.exists(video_path):
        logger.error(f"Video file not found for upload: {video_path}")
        return None
    if not youtube:
        logger.error("YouTube service object is invalid. Cannot upload.")
        return None

    try:
        logger.debug("Constructing YouTube API request body...")
        request_body = {
            "snippet": {
                "title": title,
                "description": description,
                "tags": config.YT_VIDEO_TAGS, # Use tags from config
                "categoryId": config.YT_VIDEO_CATEGORY_ID
            },
            "status": {
                "privacyStatus": config.YT_VIDEO_PRIVACY_STATUS,
                "selfDeclaredMadeForKids": config.YT_SELF_DECLARED_MADE_FOR_KIDS,
            }
        }
        logger.debug(f"Request Body: {request_body}")

        logger.info("Creating MediaFileUpload object...")
        media_file = MediaFileUpload(video_path, chunksize=-1, resumable=True, mimetype='video/*')

        logger.info("Initiating YouTube video insert request...")
        request = youtube.videos().insert(
            part="snippet,status",
            body=request_body,
            media_body=media_file
        )

        logger.info("Starting resumable upload...")
        response = None
        upload_start_time = time.time()
        last_progress_pct = -1
        while response is None:
            try:
                status, response = request.next_chunk()
                if status:
                    progress_pct = int(status.progress() * 100)
                    # Only print progress updates when the percentage changes
                    if progress_pct != last_progress_pct:
                         logger.info(f"Upload progress: {progress_pct}%")
                         last_progress_pct = progress_pct
                # Add a small sleep to avoid busy-waiting if next_chunk returns quickly without progress
                time.sleep(0.5)
            except HttpError as e:
                logger.error(f"An HTTP error occurred during upload: {e}")
                if e.resp.status in [404]:
                     logger.error("Received 404, upload may need to be restarted.")
                     return None # Indicate failure
                elif e.resp.status in [500, 502, 503, 504]:
                     logger.warning(f"Received server error {e.resp.status}, retrying might resume...")
                     time.sleep(5) # Wait longer before implicit retry by next_chunk
                else:
                     logger.error(f"Unhandled HTTP error: {e}", exc_info=True)
                     return None # Indicate failure for other HTTP errors
            except Exception as chunk_error:
                logger.error(f"An unexpected error occurred during chunk upload: {chunk_error}", exc_info=True)
                time.sleep(5) # Wait before next attempt

        upload_end_time = time.time()
        logger.info(f"Upload completed in {upload_end_time - upload_start_time:.2f} seconds.")

        video_id = response.get("id")
        if video_id:
            logger.info(f"--- YouTube Upload Successful! ---")
            logger.info(f"Video ID: {video_id}")
            logger.info(f"Watch URL: https://www.youtube.com/watch?v={video_id}")
            return video_id
        else:
            logger.error("Upload completed, but no video ID was found in the response.")
            logger.error(f"Full API Response: {response}")
            return None

    except HttpError as e:
         logger.critical(f"A critical HTTP error occurred: {e}", exc_info=True)
         return None
    except Exception as e:
        logger.critical(f"An unexpected error occurred during the YouTube upload process: {e}", exc_info=True)
        return None


# --- System / Utility Functions ---

def cleanup_temp_dir():
    global temp_dir, logger
    if temp_dir and os.path.exists(temp_dir):
        logger.info(f"Cleaning up temporary directory: {temp_dir}")
        # Attempt removal multiple times with delays, common for OS locking issues
        for attempt in range(3):
            try:
                shutil.rmtree(temp_dir)
                logger.info("Temporary directory removed successfully.")
                temp_dir = "" # Clear global var after successful removal
                return # Exit function on success
            except PermissionError as pe:
                 logger.warning(f"Cleanup Attempt {attempt+1} failed (PermissionError): {pe}. Check for open files/processes.")
                 time.sleep(0.5 + attempt * 0.5) # Increasing delay
            except OSError as oe:
                 logger.warning(f"Cleanup Attempt {attempt+1} failed (OSError): {oe}.")
                 time.sleep(0.5 + attempt * 0.5)
            except Exception as e:
                logger.warning(f"Cleanup Attempt {attempt+1} failed: {e}. Retrying...")
                time.sleep(0.5 + attempt * 0.5) # Wait longer between attempts
        # If all attempts fail
        logger.error(f"Failed to remove temporary directory '{temp_dir}' after multiple attempts.")
    else:
        logger.debug("Temporary directory cleanup skipped (path not set or directory doesn't exist).")

def load_usernames_from_file(config_obj: Config, logger_obj: logging.Logger):
    """Loads usernames from the specified file into the config."""
    logger_obj.debug(f"Attempting to load usernames from: {config_obj.USERNAME_FILE_PATH}")
    loaded_usernames = []
    try:
        # Ensure the directory for the username file exists
        user_dir = os.path.dirname(config_obj.USERNAME_FILE_PATH)
        safe_create_dir(user_dir) # Use safe_create_dir

        # Check if file exists and is not empty
        if os.path.exists(config_obj.USERNAME_FILE_PATH) and os.path.getsize(config_obj.USERNAME_FILE_PATH) > 0:
            with open(config_obj.USERNAME_FILE_PATH, 'r', encoding='utf-8') as f:
                # Read lines, strip whitespace, add 'u/' prefix, filter empty lines
                loaded_usernames = [f"u/{name.strip()}" for name in f if name.strip()]

            if loaded_usernames:
                logger_obj.info(f"Successfully loaded {len(loaded_usernames)} usernames from '{config_obj.USERNAME_FILE_PATH}'.")
            else:
                # File exists but contained no valid names after stripping
                logger_obj.warning(f"Username file found but appears empty or contains only whitespace: '{config_obj.USERNAME_FILE_PATH}'. Using default random names.")
        else:
            # File doesn't exist or is empty
            logger_obj.warning(f"Username file is missing or empty: '{config_obj.USERNAME_FILE_PATH}'. Using default random names. Creating empty file if missing.")
            # Attempt to create the file if it doesn't exist, so user knows where to put it
            if not os.path.exists(config_obj.USERNAME_FILE_PATH):
                 try:
                     with open(config_obj.USERNAME_FILE_PATH, 'a', encoding='utf-8') as f: pass
                 except Exception as create_err: logger_obj.warning(f"Could not create empty username file: {create_err}")

    except Exception as e:
        logger_obj.error(f"An error occurred while loading usernames from '{config_obj.USERNAME_FILE_PATH}': {e}", exc_info=True)
        logger_obj.warning("Proceeding with default random usernames due to error.")

    # Update the config only if usernames were successfully loaded
    if loaded_usernames:
        config_obj.HEADER_USERNAME_OPTIONS = loaded_usernames
    else:
        # If loading failed or file was empty, log that defaults are being used
        logger_obj.debug(f"Keeping default random usernames ({len(config_obj.HEADER_USERNAME_OPTIONS)} options).")


# --- Main Execution Function ---
def generate_short():
    global temp_dir, run_identifier, config, logger
    start_time_total = time.time()
    run_identifier = datetime.now().strftime("%Y%m%d_%H%M%S_%f") # Use high precision identifier
    temp_dir = os.path.join(config.TEMP_DIR_BASE, f"run_{run_identifier}")
    file_handler = None # Initialize file handler variable
    exit_code = EXIT_CODE_GENERAL_ERROR # Default to general error

    # --- Setup File Logging ---
    # (Keep file logging setup as is)
    try:
        safe_create_dir(config.OUTPUT_DIR) # Ensure output dir exists first
        log_file_path = os.path.join(config.OUTPUT_DIR, f"run_{run_identifier}.log")
        # Use 'a' mode to append if file somehow exists, utf-8 encoding
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8', mode='a')
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
        # Log initial messages to both console and file now
        logger.info(f"File logging started: {log_file_path}")
    except Exception as log_setup_err:
        logger.error(f"Failed to set up file logging to '{config.OUTPUT_DIR}': {log_setup_err}")
        # Continue without file logging if setup fails

    logger.info(f"===== Starting Short Generation Run: {run_identifier} =====")
    logger.info(f"Using Temporary Directory: {temp_dir}")
    logger.info(f"Config Snapshot: Speed={config.AUDIO_SPEED_MULTIPLIER}, Font(Main/Hdr)={config.FONT_MAIN_SIZE}/{config.FONT_HEADER_SIZE}, MaxDur={config.MAX_DURATION_SECONDS}s, EndPad={config.END_PADDING_SECONDS}s, Music={config.MUSIC_ENABLED}, BGMode={'Long' if config.USE_LONG_BACKGROUND_VIDEO else 'Short'}")
    logger.info(f"YouTube Upload Enabled: {config.ENABLE_YOUTUBE_UPLOAD}")

    # --- Path Verification ---
    # (Keep path verification as is, but it will now use the appropriate exit code)
    try:
        logger.info("Verifying essential paths and directories...")
        safe_create_dir(temp_dir) # Create the unique temp dir for this run
        safe_create_dir(config.INPUT_DIR)
        safe_create_dir(os.path.dirname(config.TEXT_FILE_PATH)) # Ensure text input dir exists
        # Background Video Path Checks
        if config.USE_LONG_BACKGROUND_VIDEO:
            if not os.path.isfile(config.LONG_BACKGROUND_VIDEO_PATH):
                raise FileNotFoundError(f"Required long background video file not found: {config.LONG_BACKGROUND_VIDEO_PATH}")
        else: # Short video mode
            safe_create_dir(config.BACKGROUND_VID_DIR_SHORT)
            # Check if directory contains any valid video files (optional but helpful)
            try:
                 if not any(os.path.splitext(f)[1].lower() in ['.mp4', '.mov', '.avi', '.mkv', '.webm'] for f in os.listdir(config.BACKGROUND_VID_DIR_SHORT) if os.path.isfile(os.path.join(config.BACKGROUND_VID_DIR_SHORT, f))):
                      logger.warning(f"Short background video directory '{config.BACKGROUND_VID_DIR_SHORT}' contains no recognized video files.")
            except OSError as list_err: logger.warning(f"Could not list files in short BG dir '{config.BACKGROUND_VID_DIR_SHORT}': {list_err}")

        safe_create_dir(config.MUSIC_DIR)
        safe_create_dir(config.USED_TEXT_DIR)
        safe_create_dir(os.path.dirname(config.USERNAME_FILE_PATH)) # Ensure usernames dir exists
        if config.AVATAR_ENABLED and config.AVATAR_PATH:
            safe_create_dir(os.path.dirname(config.AVATAR_PATH)) # Ensure avatar dir exists
            if not os.path.isfile(config.AVATAR_PATH):
                 logger.warning(f"Avatar is enabled, but image file not found at '{config.AVATAR_PATH}'. Avatar will be skipped.")

        # Check External Tool Paths
        if not os.path.isfile(config.COQUI_SCRIPT_PATH): raise FileNotFoundError(f"Coqui TTS script not found: {config.COQUI_SCRIPT_PATH}")
        if not os.path.isfile(config.COQUI_REFERENCE_AUDIO): raise FileNotFoundError(f"Coqui TTS reference audio not found: {config.COQUI_REFERENCE_AUDIO}")
        if not os.path.isdir(config.COQUI_ESPEAK_PATH): raise NotADirectoryError(f"Coqui eSpeak directory is invalid: {config.COQUI_ESPEAK_PATH}")

        # Check YouTube paths if upload enabled
        if config.ENABLE_YOUTUBE_UPLOAD:
             safe_create_dir(config.YT_DATA_DIR_PATH) # Create the YT data dir
             # Titles/Descriptions files are checked inside get_random_line now
             if not os.path.isfile(config.YT_CLIENT_SECRETS_FILE):
                  # This is critical for upload
                  raise FileNotFoundError(f"YouTube client secrets file not found: {config.YT_CLIENT_SECRETS_FILE}. Upload cannot proceed.")


        logger.info("Essential paths and directories verified.")
    except (FileNotFoundError, NotADirectoryError, ValueError, OSError) as setup_err:
        logger.critical(f"Setup failed - Critical path error: {setup_err}", exc_info=True)
        exit_code = EXIT_CODE_PRE_RUN_FAILURE # Set specific exit code
        # Cleanup and exit handled in finally block
        return exit_code # Return exit code instead of calling sys.exit here

    # --- Initialize variables ---
    final_video_clip = bg_clip = tts_clip = music_clip = audio_mix = overlay_clip = raw_tts_clip = None
    tts_path = output_path = "" # Initialize output_path here
    story_text = remaining_text = original_story_text = ""
    generation_success = False # Flag for video file creation
    upload_success = False # Flag for YT upload

    try:
        # === Step 1: Get & Clean Story Text + STORY COUNT CHECK ===
        logger.info(f"Reading and processing story from: {config.TEXT_FILE_PATH}")
        if not os.path.exists(config.TEXT_FILE_PATH):
            raise FileNotFoundError(f"Input text file not found: {config.TEXT_FILE_PATH}") # Let pre-run check handle this? No, better here.

        stories = []
        try:
            with open(config.TEXT_FILE_PATH, 'r', encoding='utf-8') as f:
                content = f.read()
            # Split into stories (paragraphs separated by double newlines) and filter empty ones
            stories = [s.strip() for s in re.split(r'\n\s*\n+', content.strip()) if s.strip()]

            # --- STORY COUNT CHECK ---
            num_stories = len(stories)
            if num_stories == 0:
                logger.critical(f"CRITICAL: No stories found in '{config.TEXT_FILE_PATH}'. Cannot generate video.")
                logger.critical(f"Please refill '{config.TEXT_FILE_PATH}' with stories.")
                exit_code = EXIT_CODE_NO_STORIES
                return exit_code # Exit early
            elif 1 <= num_stories <= 10:
                 logger.warning(f"LOW STORY COUNT: Only {num_stories} stories remaining in '{config.TEXT_FILE_PATH}'.")

            logger.info(f"Processing story 1 of {num_stories}.")
            # --- End Story Count Check ---

            original_story_text = stories[0] # Take the first story
            logger.debug(f"Original story snippet: '{original_story_text[:100]}...'")

            # Clean the selected story text for TTS
            logger.info("Cleaning story text for TTS (removing quotes, normalizing dashes)...")
            replacements = {'"': '', "'": '', "’": '', "`": '', '‘': '', '–': '-', '—': '-'}
            story_text = original_story_text
            for char, replacement in replacements.items(): story_text = story_text.replace(char, replacement)
            story_text = re.sub(r'\s+', ' ', story_text).strip() # Normalize whitespace
            logger.debug(f"Cleaned story text snippet: '{story_text[:100]}...'")

            # Prepare the remaining text to write back later
            remaining_text = "\n\n".join(stories[1:])
            # Log remaining count *after* processing current one
            logger.info(f"{len(stories)-1} stories will remain after this run.")

        except FileNotFoundError as fnf_err: # Catch specific error if file disappears between check and read
             logger.critical(f"Input text file error: {fnf_err}", exc_info=True)
             exit_code = EXIT_CODE_PRE_RUN_FAILURE
             return exit_code
        except Exception as e:
            logger.error(f"Error reading or processing stories from '{config.TEXT_FILE_PATH}': {e}", exc_info=True)
            raise RuntimeError("Failed to read or prepare story text.") from e # Re-raise to be caught below


        # === Step 1b: Generate TTS Audio ===
        # ... (no changes needed here) ...
        logger.info("Starting Coqui TTS audio generation...")
        # Ensure the text passed to TTS is the cleaned version
        cleaned_tts_text = story_text
        # Define absolute path for the raw TTS output in the temp directory
        tts_output_path = os.path.abspath(os.path.join(temp_dir, f"{run_identifier}_tts_raw.wav"))
        # Construct the command line arguments for the Coqui TTS script
        command = [
            sys.executable, # Use the current Python interpreter
            os.path.abspath(config.COQUI_SCRIPT_PATH),
            "--text", cleaned_tts_text,
            "--output_wav", tts_output_path,
            "--reference_wav", os.path.abspath(config.COQUI_REFERENCE_AUDIO),
            "--lang", config.COQUI_LANG,
            "--espeak_path", os.path.abspath(config.COQUI_ESPEAK_PATH),
             # Add other Coqui options if needed, e.g., "--use_cuda", "true"
        ]
        logger.debug(f"Coqui TTS Command: {' '.join(command)}") # Log the command for debugging

        try:
            start_tts = time.time()
            # Execute the TTS script as a subprocess
            process = subprocess.run(
                command,
                check=True, # Raise CalledProcessError if script exits with non-zero code
                capture_output=True, # Capture stdout and stderr
                text=True, # Decode stdout/stderr as text
                encoding='utf-8', # Specify encoding
                timeout=config.TTS_TIMEOUT_SECONDS # Set a timeout
            )
            end_tts = time.time()
            logger.info(f"Coqui TTS script finished execution in {end_tts - start_tts:.2f} seconds.")
            # Log stdout/stderr from the TTS script for debugging
            if process.stdout: logger.debug(f"--- Coqui TTS stdout ---\n{process.stdout.strip()}\n--- End stdout ---")
            if process.stderr: logger.info(f"--- Coqui TTS stderr ---\n{process.stderr.strip()}\n--- End stderr ---") # Use INFO for stderr as Coqui often logs progress here

            # Verify that the output file was created and is not empty
            if not os.path.isfile(tts_output_path) or os.path.getsize(tts_output_path) == 0:
                raise FileNotFoundError(f"TTS output file missing or empty after execution: '{tts_output_path}'")

            tts_path = tts_output_path # Store the path to the generated audio
            logger.info(f"TTS audio generated successfully: {os.path.basename(tts_path)}")

        except subprocess.TimeoutExpired:
            logger.critical(f"Coqui TTS script timed out after {config.TTS_TIMEOUT_SECONDS} seconds.")
            raise RuntimeError("TTS generation process timed out.")
        except subprocess.CalledProcessError as e:
            logger.critical(f"Coqui TTS script failed with return code {e.returncode}.")
            logger.error(f"stdout:\n{e.stdout}\nstderr:\n{e.stderr}") # Log captured output on error
            raise RuntimeError("TTS generation script returned an error.")
        except Exception as e:
            logger.critical(f"An unexpected error occurred during TTS execution: {e}", exc_info=True)
            raise RuntimeError("TTS process failed unexpectedly.")

        # === Step 2: Load TTS Audio & Apply Speed Adjustment ===
        # ... (no changes needed here) ...
        logger.info(f"Loading generated TTS audio: {os.path.basename(tts_path)}")
        if not tts_path or not os.path.exists(tts_path):
            raise FileNotFoundError("TTS audio file path is invalid or file missing after generation step.")
        try:
             # Load the raw TTS audio file
             raw_tts_clip = AudioFileClip(tts_path)
             raw_tts_duration = raw_tts_clip.duration
             if raw_tts_duration is None or raw_tts_duration <= 0:
                 raise ValueError(f"Loaded raw TTS audio has invalid duration: {raw_tts_duration}")
             logger.info(f"Raw TTS audio loaded. Duration: {raw_tts_duration:.3f}s")

             # Apply speed effect if multiplier is different from 1.0
             if abs(config.AUDIO_SPEED_MULTIPLIER - 1.0) > 0.01: # Use tolerance for float comparison
                 logger.info(f"Applying audio speed multiplier: {config.AUDIO_SPEED_MULTIPLIER}x")
                 # Use MoviePy's speedx effect
                 tts_clip = raw_tts_clip.fx(vfx.speedx, config.AUDIO_SPEED_MULTIPLIER)
                 # Calculate expected duration after speed change
                 adjusted_tts_duration = raw_tts_duration / config.AUDIO_SPEED_MULTIPLIER
                 # Explicitly set duration as speedx might have small inaccuracies
                 tts_clip = tts_clip.set_duration(adjusted_tts_duration)
                 logger.info(f"Audio speed adjusted. New estimated duration: {adjusted_tts_duration:.3f}s")
             else:
                 # If speed multiplier is 1.0, just use the raw clip
                 logger.info("Audio speed multiplier is 1.0. Using original TTS audio.")
                 tts_clip = raw_tts_clip
                 raw_tts_clip = None # Avoid closing the same clip twice later
                 adjusted_tts_duration = raw_tts_duration

             # Validate duration after potential adjustment
             if tts_clip.duration is None or tts_clip.duration <= 0:
                 raise ValueError("TTS audio duration became invalid after speed adjustment.")
             tts_final_duration = tts_clip.duration # This is the duration we'll work with

        except Exception as e:
            logger.error(f"Error loading or applying speed to TTS audio: {e}", exc_info=True)
            safe_close(raw_tts_clip) # Ensure cleanup
            safe_close(tts_clip)
            raise RuntimeError("Could not load or process the generated TTS audio.")

        # === Step 3: Determine Final Video Duration ===
        # ... (no changes needed here) ...
        # Calculate desired duration (TTS + padding)
        desired_duration = tts_final_duration + config.END_PADDING_SECONDS
        logger.info(f"TTS duration: {tts_final_duration:.3f}s | End Padding: {config.END_PADDING_SECONDS}s | Desired total duration: {desired_duration:.3f}s")
        # Cap the duration at the configured maximum
        final_duration = min(desired_duration, config.MAX_DURATION_SECONDS)
        logger.info(f"Final video duration set to {final_duration:.3f}s (capped at {config.MAX_DURATION_SECONDS:.2f}s).")

        # Check if TTS needs trimming to fit within the final duration minus padding
        max_allowed_tts_duration = max(0.01, final_duration - config.END_PADDING_SECONDS)
        if tts_final_duration > max_allowed_tts_duration + 0.01: # Add tolerance
             logger.warning(f"TTS audio duration ({tts_final_duration:.3f}s) exceeds the allowed time ({max_allowed_tts_duration:.3f}s) within the final video duration. Trimming TTS audio.")
             try:
                 # Trim the TTS clip to the maximum allowed duration
                 tts_clip = tts_clip.subclip(0, max_allowed_tts_duration)
                 tts_final_duration = tts_clip.duration # Update the final TTS duration
                 logger.info(f"TTS audio trimmed successfully. New duration: {tts_final_duration:.3f}s.")
             except Exception as e:
                 logger.error(f"Failed to trim TTS audio: {e}. Proceeding with original TTS duration which might cause timing issues.")
                 # Keep original tts_clip and tts_final_duration if trimming fails

        # Ensure the final TTS clip duration is explicitly set
        tts_clip = tts_clip.set_duration(tts_final_duration)


        # === Step 4 & 5: Transcribe (Whisper) & Adjust Timestamps ===
        # ... (no changes needed here) ...
        logger.info("Starting Whisper transcription on the original (pre-speed adjustment) TTS audio...")
        word_timestamps_raw = transcribe_with_whisper(tts_path) # Pass the original .wav path
        if not word_timestamps_raw:
            # If Whisper fails, we can't create timed overlays. Critical error.
            raise RuntimeError("Whisper transcription failed or returned no valid timestamps. Cannot proceed with text overlay timing.")

        logger.info(f"Adjusting {len(word_timestamps_raw)} raw word timestamps using speed multiplier ({config.AUDIO_SPEED_MULTIPLIER}x)...")
        adjusted_timestamps = []
        # Adjust each timestamp according to the speed multiplier
        for ts in word_timestamps_raw:
            # Divide start/end times by the multiplier
            adj_start = ts['start'] / config.AUDIO_SPEED_MULTIPLIER
            adj_end = ts['end'] / config.AUDIO_SPEED_MULTIPLIER
            # Ensure the adjusted timestamp start is within the *final* (potentially trimmed) TTS duration
            if adj_start < tts_final_duration:
                # Add the adjusted timestamp, ensuring end time doesn't exceed final duration
                adjusted_timestamps.append({
                    **ts, # Keep original word and confidence
                    'start': adj_start,
                    'end': min(adj_end, tts_final_duration) # Cap end time at final duration
                })
            # else: # Optional: Log timestamps that fall entirely outside the final duration
            #    logger.debug(f"Timestamp for '{ts['word']}' (adj start {adj_start:.3f}s) falls outside final TTS duration {tts_final_duration:.3f}s, excluding.")

        logger.info(f"Generated {len(adjusted_timestamps)} adjusted word timestamps relevant to the final audio duration.")
        if not adjusted_timestamps:
            # This might happen if speedup is extreme or audio is very short. Warn but continue.
            logger.warning("No word timestamps remain after speed adjustment and duration trimming. Text overlays will appear instantly at time 0.")


        # === Step 6: Prepare Background Video ===
        # ... (no changes needed here) ...
        logger.info("Preparing background video...")
        bg_clip = None
        if config.USE_LONG_BACKGROUND_VIDEO:
            # Process the specified long background video file
            logger.debug(f"Using long background video: {config.LONG_BACKGROUND_VIDEO_PATH}")
            bg_clip = process_background_video(config.LONG_BACKGROUND_VIDEO_PATH, final_duration, is_long_video=True)
        else:
            # Select a random short background video from the directory
            logger.debug(f"Selecting random short background video from: {config.BACKGROUND_VID_DIR_SHORT}")
            selected_bg_path = select_random_file(config.BACKGROUND_VID_DIR_SHORT)
            if not selected_bg_path:
                # If no video found, this is critical unless a fallback is implemented
                raise FileNotFoundError(f"No suitable background video files found in the short video directory: '{config.BACKGROUND_VID_DIR_SHORT}'.")
            logger.info(f"Using short background video: {selected_bg_path}")
            bg_clip = process_background_video(selected_bg_path, final_duration, is_long_video=False)

        # Check if background video processing was successful
        if not bg_clip:
            raise RuntimeError("Failed to load or process the background video.")


        # === Step 7: Prepare Background Music ===
        # ... (no changes needed here) ...
        music_clip = None
        if config.MUSIC_ENABLED and config.MUSIC_FILE:
            music_path = os.path.join(config.MUSIC_DIR, config.MUSIC_FILE)
            if not os.path.isfile(music_path):
                logger.warning(f"Music file specified ('{config.MUSIC_FILE}') not found in '{config.MUSIC_DIR}'. Skipping music.")
            else:
                logger.info(f"Loading and processing background music: {config.MUSIC_FILE}")
                temp_music = None # Temporary variable for loading
                try:
                    temp_music = AudioFileClip(music_path)
                    music_dur = temp_music.duration
                    if not music_dur or music_dur <= 0: raise ValueError("Music file has invalid duration.")

                    # Apply volume adjustment
                    temp_music = temp_music.volumex(config.MUSIC_VOLUME)
                    logger.debug(f"Music volume adjusted to {config.MUSIC_VOLUME}")

                    # Adjust music duration to match final video duration (loop or trim)
                    if abs(music_dur - final_duration) > 0.05: # Only adjust if significantly different
                        if music_dur < final_duration:
                            # Loop the music if it's shorter than the video
                            logger.debug(f"Music duration ({music_dur:.2f}s) is shorter than video ({final_duration:.2f}s). Looping.")
                            # Note: MoviePy loop might not be seamless, consider external tools for perfect loops if needed
                            temp_music = temp_music.loop(duration=final_duration)
                        else:
                            # Trim the music if it's longer
                            logger.debug(f"Music duration ({music_dur:.2f}s) is longer than video ({final_duration:.2f}s). Trimming.")
                            temp_music = temp_music.subclip(0, final_duration)

                    # Finalize the music clip with the correct duration
                    music_clip = temp_music.set_duration(final_duration)
                    logger.info(f"Background music prepared. Final duration: {music_clip.duration:.2f}s")

                except Exception as e:
                    logger.error(f"Error processing music file '{music_path}': {e}", exc_info=True)
                    safe_close(temp_music) # Close temporary clip if error occurs
                    music_clip = None # Ensure music_clip is None on error
        else:
            logger.info("Background music is disabled or no file specified in config.")


        # === Step 8: Generate Text Overlays ===
        # ... (no changes needed here) ...
        logger.info("Preparing text overlays with dynamic timing...")
        # Load fonts
        font_main = get_font(config.FONT_MAIN_NAME, config.FONT_MAIN_SIZE)
        font_header = get_font(config.FONT_HEADER_NAME, config.FONT_HEADER_SIZE)
        if not font_main or not font_header:
            raise RuntimeError("Failed to load required fonts for text overlays. Check font names and paths.")

        # Calculate available width for text wrapping based on margins
        max_text_wrap_width = config.OUTPUT_WIDTH - (2 * config.TEXT_MARGIN_SIDES)
        logger.info(f"Wrapping main story text to fit max width: {max_text_wrap_width}px...")
        wrapped_lines = wrap_text(story_text, font_main, max_text_wrap_width)
        if not wrapped_lines:
            # Handle case where wrapping results in no lines (e.g., empty input string)
            logger.warning("Text wrapping resulted in zero lines. No text overlays will be generated.")

        # Calculate reveal times for each wrapped line based on adjusted timestamps
        logger.info("Calculating reveal times for each text line...")
        line_start_times = calculate_line_reveal_times(wrapped_lines, adjusted_timestamps)

        # Generate the actual overlay video clip (composite of header and lines)
        logger.info("Generating the composite text overlay video clip (RGB)...")
        overlay_clip = generate_text_overlays(
            wrapped_lines,
            line_start_times,
            font_main,
            font_header,
            final_duration # Pass the final video duration
        )
        if not overlay_clip:
            # If overlay generation fails, it might be non-critical depending on requirements
            # For this script, assume it's required.
            raise RuntimeError("Failed to generate the text overlay video clip.")


        # === Step 9: Mix Audio Tracks ===
        # ... (no changes needed here) ...
        logger.info("Mixing final audio tracks...")
        clips_to_mix = [clip for clip in [tts_clip, music_clip] if clip] # Create list of valid audio clips

        if not clips_to_mix:
            # This should ideally not happen if TTS succeeded, but check anyway
            raise RuntimeError("CRITICAL: No valid audio clips available for mixing (TTS clip missing?).")
        elif len(clips_to_mix) == 1:
            # If only one track (likely just TTS), use it directly
            logger.info("Only one audio track present. Using it as the final mix.")
            audio_mix = clips_to_mix[0]
        else:
            # Mix TTS and Music using CompositeAudioClip
            logger.info(f"Mixing {len(clips_to_mix)} audio tracks (TTS and Music).")
            try:
                audio_mix = CompositeAudioClip(clips_to_mix)
                logger.info("Audio tracks mixed successfully.")
            except Exception as e:
                # Fallback to just TTS if mixing fails
                logger.error(f"Failed to composite audio tracks: {e}. Using TTS audio only as fallback.", exc_info=True)
                audio_mix = tts_clip # Fallback to the primary audio track

        # Ensure final audio mix has the correct duration and standard FPS
        try:
             audio_mix = audio_mix.set_duration(final_duration)
             current_audio_fps = getattr(audio_mix, 'fps', None)
             if not isinstance(current_audio_fps, (int, float)) or current_audio_fps <= 0 or current_audio_fps != config.STANDARD_AUDIO_FPS:
                 logger.info(f"Setting final audio mix FPS to standard {config.STANDARD_AUDIO_FPS} Hz (was {current_audio_fps}).")
                 # Setting fps attribute directly is often sufficient for MoviePy/FFmpeg
                 audio_mix.fps = config.STANDARD_AUDIO_FPS
             logger.debug(f"Final audio mix prepared. Duration: {audio_mix.duration:.3f}s, FPS: {audio_mix.fps}.")
        except Exception as e:
             logger.warning(f"Could not explicitly set final audio mix duration/FPS: {e}. Output might rely on defaults.")


        # === Step 10: Assemble Final Video ===
        # ... (no changes needed here) ...
        logger.info("Assembling final video by compositing background, overlays, and audio...")
        # Check if all required components are valid objects
        if not all([bg_clip, overlay_clip, audio_mix]):
            missing = [name for name, clip in [('Background', bg_clip), ('Overlay', overlay_clip), ('Audio Mix', audio_mix)] if not clip]
            raise RuntimeError(f"Cannot assemble final video. Missing components: {', '.join(missing)}")

        try:
            # Ensure video clips have the exact final duration
            bg_clip = bg_clip.set_duration(final_duration)
            overlay_clip = overlay_clip.set_duration(final_duration)

            # Create the final composite video
            # Place background first, then overlay on top
            # use_bgclip=True is efficient if bg_clip is the base canvas size
            final_video_clip = CompositeVideoClip(
                [bg_clip, overlay_clip],
                size=(config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT),
                use_bgclip=True # Use background as the base layer
            )

            # Set the final mixed audio to the composite video clip
            final_video_clip = final_video_clip.set_audio(audio_mix)
            # Explicitly set the duration of the final video clip
            final_video_clip = final_video_clip.set_duration(final_duration)

            logger.info(f"Final video assembled successfully. Target duration: {final_video_clip.duration:.3f}s")

        except Exception as e:
            logger.error(f"Error during final video compositing: {e}", exc_info=True)
            raise RuntimeError("Final video assembly process failed.")


        # === Step 11: Write Output Video File ===
        output_filename = f"short_{run_identifier}.mp4"
        output_path = os.path.join(config.OUTPUT_DIR, output_filename) # Assign value here
        logger.info(f"Writing final video file to: {output_path}")

        if abs(final_video_clip.duration - final_duration) > 0.05:
            logger.warning(f"Final assembled clip duration ({final_video_clip.duration:.3f}s) differs significantly from target ({final_duration:.3f}s) before writing.")

        start_render_time = time.time()
        try:
            # --- !!! QUALITY SETTINGS FINE-TUNED HERE !!! ---
            keyframe_interval = config.OUTPUT_FPS * 2 # e.g., 60 for 30fps
            ffmpeg_params = [
                '-crf', '18',                  # Keep low CRF (or try 20 again with these tunes)
                '-profile:v', 'high',
                '-level', '4.1',
                '-pix_fmt', 'yuv420p',
                '-tune', 'film',               # *** Added tune parameter ***
                '-g', str(keyframe_interval),  # *** Added keyframe interval ***
                '-b:a', '320k'                 # *** Added audio bitrate ***
            ]
            temp_audio_path = os.path.join(temp_dir, f'temp_audio_{run_identifier}.aac')

            logger.info(f"Starting video render with settings: CRF=18, Preset=slow, Tune=film, Keyframe Interval={keyframe_interval}, Audio Bitrate=320k")

            # Write the video file using moviepy's write_videofile
            final_video_clip.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",             # AAC is generally preferred by YouTube over mp3
                temp_audiofile=temp_audio_path,
                remove_temp=True,
                preset="slow",                 # Keep slow preset
                ffmpeg_params=ffmpeg_params,
                threads=max(1, os.cpu_count() // 2 if os.cpu_count() else 2),
                logger='bar',
                fps=config.OUTPUT_FPS,
                audio_bitrate="320k"           # Also specify here for clarity/redundancy
            )
            # --- !!! END OF QUALITY SETTINGS UPDATE !!! ---

            end_render_time = time.time()
            logger.info(f"Video rendering completed in {end_render_time - start_render_time:.2f} seconds.")

            if not os.path.exists(output_path) or os.path.getsize(output_path) < 10000:
                raise RuntimeError(f"Video file writing process completed, but the output file '{output_path}' is missing or unexpectedly small.")

            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"Final video saved successfully: {output_path} (Size: {file_size_mb:.2f} MB)")
            generation_success = True

        except Exception as e:
            logger.error(f"Failed to write the final video file: {e}", exc_info=True)
            if os.path.exists(output_path):
                try: os.remove(output_path); logger.info(f"Removed potentially incomplete output file: {output_path}")
                except Exception as rm_err: logger.warning(f"Could not remove failed output file '{output_path}': {rm_err}")
            generation_success = False
            raise RuntimeError("Video writing process failed.")


        # === Step 12: Update Text Files (if generation succeeded) ===
        # ... (no changes needed here) ...
        if generation_success: # Use generation_success flag
            logger.info("Video generation successful. Updating text files...")
            try:
                # Ensure the directory for the used text file exists
                safe_create_dir(os.path.dirname(config.USED_TEXT_FILE))
            except Exception as e:
                logger.warning(f"Could not ensure used text directory exists: {e}. Archiving might fail.")

            # Archive the used story
            try:
                # Use the original story text for archiving to preserve formatting etc.
                text_to_archive = original_story_text
            except NameError:
                 logger.error("'original_story_text' not defined for archiving. Using cleaned 'story_text'.")
                 text_to_archive = story_text # Fallback to cleaned text

            try:
                with open(config.USED_TEXT_FILE, 'a', encoding='utf-8') as f_used:
                    f_used.write(f"--- Story Used: Run {run_identifier} ---\n")
                    f_used.write(text_to_archive)
                    f_used.write("\n\n") # Add separation
                logger.info(f"Appended used story to: {config.USED_TEXT_FILE}")
            except Exception as e:
                logger.error(f"Failed to append used story to '{config.USED_TEXT_FILE}': {e}")

            # Overwrite the input file with remaining stories
            try:
                with open(config.TEXT_FILE_PATH, 'w', encoding='utf-8') as f_input:
                    f_input.write(remaining_text)
                # Use num_stories defined earlier for accurate count before update
                logger.info(f"Overwrote input file '{config.TEXT_FILE_PATH}' with remaining stories ({num_stories-1} left).")
            except Exception as e:
                logger.error(f"Failed to overwrite input text file '{config.TEXT_FILE_PATH}' with remaining stories: {e}")
        else:
            logger.warning("Video generation failed. Skipping text file updates.")


        # === Step 13: YouTube Upload (if generation succeeded and upload enabled) ===
        if generation_success and config.ENABLE_YOUTUBE_UPLOAD:
            logger.info("--- Starting YouTube Upload Process ---")
            upload_success = False # Reset flag for this section
            try:
                # 1. Get Title and Description Parts
                title_base = get_random_line(config.YT_TITLE_FILE_PATH)
                description_base = get_random_line(config.YT_DESCRIPTION_FILE_PATH)
                part_number = get_unique_part_number()

                # Handle defaults if files are missing/empty or part generation fails
                if title_base is None:
                     logger.warning("Could not get title base from file. Using default.")
                     title_base = "Reddit Story Short" # Or another suitable default
                if description_base is None:
                     logger.warning("Could not get description base from file. Using default.")
                     description_base = "" # Default to empty description, or add standard text
                if part_number == 0:
                     logger.warning("Could not generate a unique part number. Using random fallback for title.")
                     part_number = random.randint(1000, 9999) # Use a non-conflicting random number

                # 2. Construct Full Title/Description (Applying user requests)
                title = f"{title_base} - Part {part_number}"
                description = description_base # Use only the base description

                logger.info(f"Generated Video Title: {title}")
                logger.info(f"Using Video Description from file: '{description[:100]}...'")
                logger.info(f"Using Video Tags from config: {config.YT_VIDEO_TAGS}")

                # 3. Authenticate
                youtube_service = authenticate_youtube()

                # 4. Upload (output_path was defined in Step 11)
                if youtube_service:
                    video_id = upload_video(youtube_service, output_path, title, description)
                    if video_id:
                        upload_success = True
                    else:
                        logger.error("YouTube upload function completed but did not return a video ID.")
                        exit_code = EXIT_CODE_UPLOAD_FAILURE # Set specific failure code
                else:
                    logger.error("Failed to authenticate with YouTube. Cannot upload video.")
                    exit_code = EXIT_CODE_UPLOAD_FAILURE # Set specific failure code

            except Exception as upload_err:
                logger.critical(f"An unexpected error occurred during the YouTube upload process: {upload_err}", exc_info=True)
                upload_success = False # Ensure flag is false on error
                exit_code = EXIT_CODE_UPLOAD_FAILURE # Set specific failure code

            if upload_success:
                 logger.info("--- YouTube Upload Process Completed Successfully ---")
                 exit_code = EXIT_CODE_SUCCESS # Set success code only if upload worked
            else:
                 logger.error("--- YouTube Upload Process Failed ---")
                 # Keep the exit code set previously (likely UPLOAD_FAILURE)
        elif generation_success and not config.ENABLE_YOUTUBE_UPLOAD:
             logger.info("YouTube upload is disabled in the configuration. Skipping upload.")
             exit_code = EXIT_CODE_SUCCESS # Generation succeeded, upload skipped = overall success for this run

        # --- Final Success Message ---
        if generation_success and (upload_success or not config.ENABLE_YOUTUBE_UPLOAD):
            logger.info("===== Short Generation Process Completed Successfully =====")
            exit_code = EXIT_CODE_SUCCESS
        elif generation_success and config.ENABLE_YOUTUBE_UPLOAD and not upload_success:
             logger.error("===== Short Generation Succeeded, BUT Upload Failed =====")
             exit_code = EXIT_CODE_UPLOAD_FAILURE # Prioritize upload failure code


    # --- Error Handling for the Main Process ---
    except (FileNotFoundError, NotADirectoryError) as e:
        logger.error(f"!!! GENERATION FAILED: A required file or directory was not found: {e}", exc_info=True)
        exit_code = EXIT_CODE_PRE_RUN_FAILURE # Or GENERATION_FAILURE if it happened mid-process? Let's use specific.
        # Check where error happened? More complex. Stick to GENERATION_FAILURE for now if not pre-run
        if 'setup_err' not in locals(): # If it's not the initial setup error
            exit_code = EXIT_CODE_GENERATION_FAILURE

    except RuntimeError as e: # Catch specific runtime errors raised in the process
        logger.error(f"!!! GENERATION FAILED: A critical process step failed: {e}", exc_info=True)
        exit_code = EXIT_CODE_GENERATION_FAILURE
    except Exception as e: # Catch any other unexpected exceptions
        logger.critical("!!! GENERATION FAILED: An unexpected critical error occurred:", exc_info=True)
        exit_code = EXIT_CODE_GENERAL_ERROR

    # --- Final Cleanup ---
    finally:
        logger.info("--- Starting Final Cleanup Phase ---")
        logger.debug("Closing any open MoviePy clips...")
        # Close all potentially open MoviePy clips safely
        safe_close(final_video_clip)
        safe_close(overlay_clip)
        safe_close(bg_clip)
        safe_close(tts_clip)
        safe_close(music_clip)
        safe_close(audio_mix)
        # Ensure raw_tts_clip is closed if it's a separate object
        if 'raw_tts_clip' in locals() and raw_tts_clip and raw_tts_clip is not tts_clip:
            safe_close(raw_tts_clip)
        logger.info("MoviePy resources closed.")

        # Decide whether to keep or remove the temporary directory
        # Keep temp dir only if generation failed, not if just upload failed
        if not generation_success and exit_code != EXIT_CODE_NO_STORIES:
             logger.warning(f"Run FAILED (Code: {exit_code}). Temporary directory will be kept for debugging: {temp_dir}")
        else:
            cleanup_temp_dir() # Remove temp files on success or non-generation failures

        # Log total execution time
        end_time_total = time.time()
        logger.info(f"Total script execution time: {end_time_total - start_time_total:.2f} seconds")
        logger.info(f"===== Run Finished: {run_identifier} (Exit Code: {exit_code}) =====")
        logger.info("="*70 + "\n") # Separator for logs

        # Close the file handler for logging
        if file_handler:
            try:
                logger.info("Closing log file handler.")
                file_handler.close()
                logger.removeHandler(file_handler) # Remove handler to avoid issues if script is re-run
            except Exception as e:
                # Log closing error to console if possible
                logger.error(f"Error closing log file handler: {e}")

        # Return the final exit code
        return exit_code


# --- Script Entry Point ---
if __name__ == "__main__":
    logger.info("Script execution started. Performing pre-run checks...")
    final_exit_code = EXIT_CODE_GENERAL_ERROR # Default assumption
    checks_passed = True

    # --- Essential File/Directory Checks ---
    # (Keep pre-run checks as they are)
    essentials = {
        "Coqui Script": (config.COQUI_SCRIPT_PATH, 'file'),
        "Coqui Reference Audio": (config.COQUI_REFERENCE_AUDIO, 'file'),
        "Coqui eSpeak Path": (config.COQUI_ESPEAK_PATH, 'dir'),
        "Input Text File": (config.TEXT_FILE_PATH, 'file'),
    }
    # Add background check based on mode
    if config.USE_LONG_BACKGROUND_VIDEO:
        essentials["Long Background Video"] = (config.LONG_BACKGROUND_VIDEO_PATH, 'file')
    else:
         essentials["Short Background Dir"] = (config.BACKGROUND_VID_DIR_SHORT, 'dir')

    # Add YouTube specific checks if enabled
    if config.ENABLE_YOUTUBE_UPLOAD:
        # Client Secrets is critical for upload to even attempt auth
        essentials["YouTube Client Secrets"] = (config.YT_CLIENT_SECRETS_FILE, 'file')
        # Non-fatal checks for supporting YouTube files (will use defaults if missing)
        if not os.path.isfile(config.YT_TITLE_FILE_PATH): logger.warning(f"PRE-CHECK WARNING: YouTube title file missing: {config.YT_TITLE_FILE_PATH}. Upload will use fallback title.")
        if not os.path.isfile(config.YT_DESCRIPTION_FILE_PATH): logger.warning(f"PRE-CHECK WARNING: YouTube description file missing: {config.YT_DESCRIPTION_FILE_PATH}. Upload will use fallback description.")

    for name, (path, check_type) in essentials.items():
        if check_type == 'file' and not os.path.isfile(path):
            logger.critical(f"FATAL PRE-CHECK FAILED: Required file missing: '{name}' at {path}")
            checks_passed = False
        elif check_type == 'dir' and not os.path.isdir(path):
            logger.critical(f"FATAL PRE-CHECK FAILED: Required directory missing: '{name}' at {path}")
            checks_passed = False

    # --- Non-Fatal Warnings ---
    # (Keep non-fatal checks as they are)
    # Check if text file is empty
    if os.path.isfile(config.TEXT_FILE_PATH):
        try:
            if os.path.getsize(config.TEXT_FILE_PATH) == 0:
                 logger.warning(f"PRE-CHECK WARNING: Input text file '{config.TEXT_FILE_PATH}' exists but is empty.")
        except OSError as e: logger.warning(f"PRE-CHECK WARNING: Could not check size of text file '{config.TEXT_FILE_PATH}': {e}")

    # Check music file if enabled
    if config.MUSIC_ENABLED and config.MUSIC_FILE:
        music_full_path = os.path.join(config.MUSIC_DIR, config.MUSIC_FILE)
        if not os.path.isfile(music_full_path):
            logger.warning(f"PRE-CHECK WARNING: Music is enabled, but file '{config.MUSIC_FILE}' not found in '{config.MUSIC_DIR}'.")

    # Check avatar file if enabled
    if config.AVATAR_ENABLED and config.AVATAR_PATH and not os.path.isfile(config.AVATAR_PATH):
        logger.warning(f"PRE-CHECK WARNING: Avatar is enabled, but file not found at '{config.AVATAR_PATH}'.")

    # Check Username file (non-fatal)
    if not os.path.isfile(config.USERNAME_FILE_PATH):
        logger.info(f"PRE-CHECK INFO: Username file missing: {config.USERNAME_FILE_PATH}. Default random names will be used.")


    # --- Execute Main Function or Exit ---
    if checks_passed:
        # Load usernames only if essential checks passed
        load_usernames_from_file(config, logger)
        logger.info("Pre-run checks passed (or only warnings were issued). Starting main generation process...")
        # Run generate_short and get the exit code it determined
        final_exit_code = generate_short()

    else:
        logger.critical("One or more FATAL pre-run checks failed. Aborting execution.")
        final_exit_code = EXIT_CODE_PRE_RUN_FAILURE

    # Ensure log file handler is closed if it was opened
    # (This might be redundant if generate_short already closed it, but safe)
    f_handler = next((h for h in logger.handlers if isinstance(h, logging.FileHandler)), None)
    if f_handler:
        try: f_handler.close(); logger.removeHandler(f_handler)
        except Exception: pass

    # Exit the script with the determined code
    sys.exit(final_exit_code)
