import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import re
import time
import unicodedata
from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL
import tempfile
import pathlib
import ffmpeg
from typing import Optional, Dict
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, TooManyRequests
import random


# Load environment variables
load_dotenv()
ARIA_BASE_URL = os.getenv('aria_base_url')
ARIA_API_KEY = os.getenv('aria_api_key')

# Check required environment variables
if not ARIA_BASE_URL or not ARIA_API_KEY:
    st.error("Please set the aria_base_url and aria_api_key environment variables in the .env file")
    st.stop()

# Configure Aria client
client = OpenAI(
    base_url=ARIA_BASE_URL,
    api_key=ARIA_API_KEY
)

# Add new session state keys for caption generation
SESSION_STATE_KEYS = [
    'processed_segments',
    'current_video_url',
    'youtube_url',
    'analysis_prompt',
    'duration',
    'transcript',
    'analysis_result',
    'downloaded_videos',
    'reset_counter',
    'page_state',
    'captions',  # New: Store captions for each segment
    'generating_caption'  # New: Track caption generation state
]

# Initialize session states
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    for key in SESSION_STATE_KEYS:
        if key == 'processed_segments':
            st.session_state[key] = []
        elif key == 'downloaded_videos':
            st.session_state[key] = {}
        elif key == 'reset_counter':
            st.session_state[key] = 0
        elif key == 'page_state':
            st.session_state[key] = {'processed': False, 'segments_displayed': False}
        elif key == 'captions':
            st.session_state[key] = {}
        elif key == 'generating_caption':
            st.session_state[key] = None
        else:
            st.session_state[key] = None


def clear_all_state():
    """Completely clear all state variables and increment reset counter"""
    # Increment reset counter
    if 'reset_counter' in st.session_state:
        st.session_state.reset_counter += 1
    else:
        st.session_state.reset_counter = 1
    
    # Clear all states except reset counter
    for key in list(st.session_state.keys()):
        if key != 'reset_counter':
            del st.session_state[key]
    
    # Reinitialize with empty values
    st.session_state.processed_segments = []
    st.session_state.current_video_url = None
    st.session_state.youtube_url = None
    st.session_state.analysis_prompt = None
    st.session_state.duration = None
    st.session_state.transcript = None
    st.session_state.analysis_result = None
    st.session_state.downloaded_videos = {}
    st.session_state.page_state = {'processed': False, 'segments_displayed': False}
    
    # Clear all caches
    st.cache_data.clear()
    st.cache_resource.clear()


def sanitize_filename(filename):
    """Sanitize filename to remove invalid characters."""
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    filename = filename.replace(' ', '_')
    filename = re.sub(r'[^\w\-_.]', '', filename)
    filename = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode()
    return filename[:200]

def get_video_id(url):
    """Extract video ID from YouTube URL."""
    if 'youtu.be' in url:
        return url.split('/')[-1].split('?')[0]
    elif 'youtube.com' in url:
        return re.search(r'v=([^&]+)', url).group(1)
    return url

def get_youtube_transcript(url):
    """Get transcript from YouTube video."""
    try:
        video_id = get_video_id(url)
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        
        formatted_transcript = []
        for entry in transcript_list:
            start_time = int(entry['start'])
            text = entry['text']
            minutes = start_time // 60
            seconds = start_time % 60
            timestamp = f"{minutes:02d}:{seconds:02d}"
            formatted_transcript.append(f"[{timestamp}] {text}")
        
        return "\n".join(formatted_transcript)
    except Exception as e:
        st.error(f"Error getting transcript: {str(e)}")
        return None

def split_transcript_by_duration(transcript, chunk_minutes=60):
    """Split transcript into chunks of specified duration."""
    chunks = []
    current_chunk = []
    current_minutes = 0
    
    for line in transcript.split('\n'):
        # Extract timestamp [MM:SS]
        match = re.match(r'\[(\d{2}):(\d{2})\]', line)
        if match:
            minutes, seconds = map(int, match.groups())
            timestamp_minutes = minutes
            
            # If we've exceeded our chunk duration, start a new chunk
            if timestamp_minutes >= (len(chunks) + 1) * chunk_minutes:
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                current_chunk = []
            
            current_chunk.append(line)
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks

def merge_segment_responses(responses):
    """Merge multiple segment responses into a single response."""
    try:
        all_segments = []
        
        for response in responses:
            # Parse each response
            cleaned_response = re.search(r'\{[\s\S]*\}', response).group()
            response_data = json.loads(cleaned_response)
            all_segments.extend(response_data['segments'])
        
        # Sort segments by start time
        all_segments.sort(key=lambda x: int(x['start_time'].split(':')[0]) * 60 + int(x['start_time'].split(':')[1]))
        
        # Create merged response
        merged_response = {
            "segments": all_segments
        }
        
        return json.dumps(merged_response, indent=2)
    except Exception as e:
        st.error(f"Error merging segment responses: {str(e)}")
        return None

def analyze_transcript_with_aria(transcript, content_type, duration_request, custom_prompt=None):
    """Analyze transcript using Aria AI with support for long videos."""
    try:
        # Common prompt construction moved outside if-else
        prompt_map = {
            'EDUCATIONAL': "Extract complete educational segments that fully explain a single concept",
            'KEY_MOMENTS': "Identify complete pivotal conversations or revelations",
            'MOTIVATIONAL': "Find complete inspirational stories or messages",
            'INSIGHTFUL': "Extract focused, complete insights or arguments",
            'CUSTOM': custom_prompt
        }
        
        selected_prompt = prompt_map.get(content_type, custom_prompt)
        
        # Duration guidance
        if duration_request == 60:
            duration_guidance = """
            Duration and Precision Requirements:
            - Target duration: 1-3 minutes per clip
            - Include ONLY the essential question-answer exchange
            - If context is referenced, include ONLY the directly relevant setup
            - Trim any tangents or unrelated discussions
            - End clip as soon as the point is fully concluded
            """
        else:
            duration_guidance = """
            Content Precision Requirements:
            - Each clip should contain ONE complete thought or discussion
            - Include ONLY the context that's necessary for understanding
            - Start at the precise moment the topic is introduced
            - End exactly when the point is fully made
            - Remove tangents or side discussions
            """
        
        # Base prompt template
        enhanced_prompt = f"""You are a precision content curator specializing in extracting complete but focused segments from long-form conversations. Analyze this transcript to find self-contained, meaningful segments that are complete yet concise.

        CORE OBJECTIVE: {selected_prompt}

        {duration_guidance}

        CLIP SELECTION CRITERIA:
        1. PRECISE COMPLETENESS:
           - Start exactly where the relevant question or context begins
           - Include the complete core answer or discussion
           - Include ONLY follow-ups that directly enhance the main point
           - End precisely when the point is fully made
           - Do NOT include tangential discussions or examples

        2. DURATION GUIDELINES:
           - Minimum 60 seconds to ensure proper context
           - Typical range: 1-3 minutes for most segments
           - Only extend beyond 3 minutes if absolutely necessary for completeness
           - End clip immediately after point is fully made

        3. CONTEXTUAL PRECISION:
           - Include ONLY the context necessary for understanding
           - If a previous point is referenced, include ONLY that specific reference
           - Cut out any side discussions or tangents
           - Remove unnecessary examples or repetitions

        4. CONTENT FOCUS:
           - Each clip should contain ONE clear main point or story
           - Ensure the segment stands alone without needing external context
           - Keep only the elements that directly contribute to the main point
           - Remove interesting but non-essential tangents

        Generate a maximum of 8 clips that meet ALL above criteria.
        If user requests a single clip, select the MOST relevant complete segment that precisely matches their request.

        Provide your response in this EXACT format:
        {{
          "segments": [
              {{
                  "start_time": "MM:SS",
                  "end_time": "MM:SS",
                  "description": "Clear title that captures the main point",
                  "twitter_caption": "Focused tweet highlighting core message",
                  "content_type": "{content_type}"
              }}
          ]
        }}
        """

        # Split transcript into 60-minute chunks
        transcript_chunks = split_transcript_by_duration(transcript)
        
        # If only one chunk, process normally
        if len(transcript_chunks) <= 1:
            final_prompt = enhanced_prompt + f"\nTranscript:\n{transcript}"
            response = client.chat.completions.create(
                model="aria",
                messages=[{"role": "user", "content": final_prompt}],
                stop=["<|im_end|>"],
                stream=False,
                temperature=0.6,
                max_tokens=1024,
                top_p=1
            )
            return response.choices[0].message.content
        
        else:
            # Process each chunk separately
            responses = []
            
            for i, chunk in enumerate(transcript_chunks):
                # Modify prompt for chunk processing
                chunk_prompt = f"""You are analyzing part {i+1} of {len(transcript_chunks)} from a longer video. 
                Focus on finding the best segments within this portion (Minutes {i*60}-{(i+1)*60}).
                
                {enhanced_prompt}
                
                Transcript Chunk {i+1}:
                {chunk}
                """
                
                # Process chunk
                chunk_response = client.chat.completions.create(
                    model="aria",
                    messages=[{"role": "user", "content": chunk_prompt}],
                    stop=["<|im_end|>"],
                    stream=False,
                    temperature=0.6,
                    max_tokens=1024,
                    top_p=1
                )
                
                responses.append(chunk_response.choices[0].message.content)
            
            # Merge responses from all chunks
            merged_response = merge_segment_responses(responses)
            return merged_response

    except Exception as e:
        st.error(f"Error analyzing transcript: {str(e)}")
        return None

def parse_segments_response(response_text):
    """Parse the structured response from Gemini."""
    try:
        cleaned_response = re.search(r'\{[\s\S]*\}', response_text).group()
        response_data = json.loads(cleaned_response)
        
        parsed_segments = []
        for segment in response_data['segments']:
            start_time_str = segment['start_time']
            end_time_str = segment['end_time']
            
            def time_to_seconds(time_str):
                minutes, seconds = map(int, time_str.split(':'))
                return minutes * 60 + seconds
            
            parsed_segments.append({
                'start_time': time_to_seconds(start_time_str),
                'end_time': time_to_seconds(end_time_str),
                'description': segment['description'],
                'twitter_caption': segment['twitter_caption'],
                'content_type': segment['content_type']
            })
        
        return parsed_segments
    except Exception as e:
        st.error(f"Error parsing segments: {str(e)}")
        return None
    
def download_video_segment(video_url, start_time, end_time, output_file):
    """
    Download 720p video segment of YouTube video using fast copy method.
    Optimized for 720p quality with maximum download efficiency.
    """
    try:
        ydl_opts = {
            # Target 720p MP4, with fallback options
            'format': 'bestvideo[height=720][ext=mp4]+bestaudio[ext=m4a]/best[height=720][ext=mp4]/best[height<=720]',
            'outtmpl': output_file,
            'external_downloader': 'ffmpeg',
            'external_downloader_args': [
                '-ss', str(start_time),
                '-to', str(end_time),
                '-c', 'copy',  # Fast copy without re-encoding
                '-maxrate', '4M',  # Adjusted bitrate for 720p
                '-bufsize', '8M'   # Adjusted buffer size for 720p
            ],
            'http_chunk_size': 5242880,  # 5MB chunks - optimized for 720p
            'retries': 10,
        }

        with YoutubeDL(ydl_opts) as ydl:
            ydl.params['buffersize'] = 16384  # 16KB buffer - suitable for 720p
            ydl.params['socket_timeout'] = 20  # Reduced timeout since files are smaller
            ydl.download([video_url])
        
        return True
    except Exception as e:
        st.error(f"Error downloading video segment: {str(e)}")
        return False
    
def get_content_type_emoji(content_type):
    """Get appropriate emoji for content type."""
    emoji_map = {
        'MOTIVATIONAL': '🔥',
        'EDUCATIONAL': '📚',
        'KEY_INSIGHT': '💡',
        'EMOTIONAL_MOMENT': '💫',
        'ACTION_ITEM': '✅'
    }
    return emoji_map.get(content_type, '📎')

def get_transcript_segment(transcript, start_time, end_time):
    """Extract transcript text between start and end timestamps."""
    lines = transcript.split('\n')
    segment_lines = []
    
    for line in lines:
        # Extract timestamp from line (format: [MM:SS])
        match = re.match(r'\[(\d{2}):(\d{2})\]', line)
        if match:
            minutes, seconds = map(int, match.groups())
            line_time = minutes * 60 + seconds
            
            # Check if line falls within our time window
            if start_time <= line_time <= end_time:
                # Remove timestamp from line
                clean_line = re.sub(r'\[\d{2}:\d{2}\]\s*', '', line)
                segment_lines.append(clean_line)
    
    return ' '.join(segment_lines)

def generate_new_caption(segment, current_caption, title, platform_prompt, transcript):
    """Generate concise, engaging social media captions based on video content"""
    try:
        segment_text = get_transcript_segment(
            transcript,
            segment['start_time'],
            segment['end_time']
        )
        
        prompt = f"""
        Generate a single engaging social media caption for this video segment:
        
        Context:
        - Title: {title}
        - Content: {segment_text}
        
        Requirements:
        1. Format: Start with a bold statement using ** ** followed by 2-3 relevant hashtags
        2. Length: Keep it between 100-150 characters total
        3. Style: Make it punchy, clear, and immediately engaging
        4. Focus: Highlight ONE key insight or moment from the segment
        5. Structure: [Bold statement] + [One supporting sentence] + [Hashtags]
        
        Example format:
        **Key insight stated boldly!** Brief follow-up that adds value. #Hashtag1 #Hashtag2

        Generate only the caption, no additional text or explanations.
        """

        response = client.chat.completions.create(
            model="aria",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100
        )
        
        caption = response.choices[0].message.content.strip()
        
        # Clean up any potential formatting issues
        if not caption.startswith("**"):
            caption = f"**{caption.split()[0]}**" + " " + " ".join(caption.split()[1:])
        
        return caption

    except Exception as e:
        st.error(f"Error generating caption: {str(e)}")
        return None

def process_and_display_segment(segment, idx, youtube_url, temp_dir):
    """Process and display a single video segment with enhanced UI and simplified caption modification feature"""
    output_file = os.path.join(temp_dir, f"segment_{idx + 1}.mp4")
    video_key = f'video_{idx}'
    
    # Initialize caption state if not exists
    if video_key not in st.session_state.captions:
        st.session_state.captions[video_key] = segment['twitter_caption']
    
    # Check if we need to download the video
    if video_key not in st.session_state.downloaded_videos:
        with st.spinner(f"Downloading segment {idx + 1}..."):
            success = download_video_segment(
                youtube_url,
                segment['start_time'],
                segment['end_time'],
                output_file
            )
            
            if success:
                with open(output_file, 'rb') as video_file:
                    st.session_state.downloaded_videos[video_key] = video_file.read()
    
    # Display styling (kept from current version)
    st.markdown("""
        <style>
        .clip-container {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .clip-title {
            color: #1f1f1f;
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 20px;
        }
        .clip-caption {
            background-color: #f8f9fa;
            border-left: 4px solid #0066cc;
            padding: 12px;
            margin-top: 0;
            margin-bottom: 10px;
            font-size: 16px;
            font-weight: 700;
            color: #2c3e50;
            border-radius: 0 5px 5px 0;
            line-height: 1.4;
        }
        .caption-input {
            margin-top: 5px;
            padding: 8px;
        }
        .caption-container {
            margin-top: 56px;
        }
        .video-info {
            font-size: 14px;
            color: #666;
            margin-bottom: 10px;
        }
        .timestamp-badge {
            background-color: #e9ecef;
            padding: 4px 8px;
            border-radius: 4px;
            margin-right: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Create container for each clip
    with st.container():
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f'<div class="clip-title">{segment["description"]}</div>', unsafe_allow_html=True)
            
            # Display timestamp information
            st.markdown(
                f'''<div class="video-info">
                    <span class="timestamp-badge">Start: {format_time(segment["start_time"])}</span>
                    <span class="timestamp-badge">End: {format_time(segment["end_time"])}</span>
                    <span class="timestamp-badge">Duration: {format_time(segment["end_time"] - segment["start_time"])}</span>
                </div>''',
                unsafe_allow_html=True
            )
            
            if video_key in st.session_state.downloaded_videos:
                video_player_key = f"video_player_{idx}_{st.session_state.reset_counter}"
                st.video(st.session_state.downloaded_videos[video_key])
            else:
                st.warning("Video segment not available")
        
        with col2:
            # Caption container
            st.markdown('<div class="caption-container">', unsafe_allow_html=True)
            st.markdown(f'<div class="clip-caption">{st.session_state.captions[video_key]}</div>', unsafe_allow_html=True)
            
            # Simplified caption modification interface
            st.markdown('<div class="caption-input">', unsafe_allow_html=True)
            caption_prompt = st.text_input(
                "✍️ Refine caption",
                placeholder="Add emojis, make it engaging...",
                key=f"caption_prompt_{idx}"
            )

            if st.button("🔄 Generate New Caption", key=f"generate_caption_{idx}"):
                with st.spinner("Generating new caption..."):
                    st.session_state.generating_caption = idx
                    new_caption = generate_new_caption(
                        segment,
                        st.session_state.captions[video_key],
                        segment['description'],
                        caption_prompt,
                        st.session_state.transcript
                    )
                    
                    if new_caption:
                        st.session_state.captions[video_key] = new_caption
                        st.rerun()
            
            st.markdown('</div></div>', unsafe_allow_html=True)
        
        # Add separator between clips
        if idx < len(st.session_state.processed_segments) - 1:
            st.markdown("<hr style='margin: 30px 0;'>", unsafe_allow_html=True)

def format_time(seconds):
    """Convert seconds to MM:SS format"""
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return f"{int(minutes):02d}:{int(remaining_seconds):02d}"

def main():
    st.set_page_config(
        page_title="YouClipAI",
        page_icon="🎥",
        layout="wide"
    )
    
    # Store previous parameter values in session state if not exists
    if 'previous_params' not in st.session_state:
        st.session_state.previous_params = {
            'youtube_url': '',
            'content_type': '',
            'duration': '',
            'custom_prompt': ''
        }
    
    # Sidebar inputs with state preservation
    with st.sidebar:
        youtube_url = st.text_input(
            "YouTube URL",
            value=st.session_state.get('youtube_url', ''),
            placeholder="https://youtube.com/watch?v=...",
            key=f"url_input_{st.session_state.reset_counter}"
        )
        
        content_type = st.selectbox(
            "What type of clips do you want to find?",
            ["EDUCATIONAL", "KEY_MOMENTS", "MOTIVATIONAL", "INSIGHTFUL", "CUSTOM"],
            help="Select the type of content you want to extract",
            key=f"content_type_{st.session_state.reset_counter}"
        )
        
        custom_prompt = None
        if content_type == "CUSTOM":
            custom_prompt = st.text_area(
                "Custom Analysis Prompt",
                value=st.session_state.get('custom_prompt', ''),
                placeholder="E.g., Find best moments from last 20 minutes...",
                key=f"custom_prompt_{st.session_state.reset_counter}"
            )
        
        duration = st.selectbox(
            "Target clip duration (seconds)",
            options=[60,120,180,'any'],
            help="Select the target duration for video segments. Choose 60 seconds for Instagram/Reels optimized clips.",
            key=f"duration_{st.session_state.reset_counter}"
        )

        # Check if any parameter has changed
        params_changed = (
            youtube_url != st.session_state.previous_params['youtube_url'] or
            content_type != st.session_state.previous_params['content_type'] or
            duration != st.session_state.previous_params['duration'] or
            (content_type == "CUSTOM" and custom_prompt != st.session_state.previous_params['custom_prompt'])
        )

        process_button = st.button(
            "🎬 Generate Clips",
            key=f"process_{st.session_state.reset_counter}"
        )

        st.markdown("<br>" * 3, unsafe_allow_html=True)
        
        if st.button(
            "🔄 New Project",
            help="Clear all current clips and start fresh with a new video",
            key=f"start_over_{st.session_state.reset_counter}"
        ):
            clear_all_state()
            st.rerun()

    # Main content area logic
    if process_button and youtube_url:
        # If parameters changed, clear all relevant state
        if params_changed:
            # Clear all processing-related states
            st.session_state.processed_segments = []
            st.session_state.downloaded_videos = {}
            st.session_state.captions = {}
            st.session_state.transcript = None
            st.session_state.analysis_result = None
            st.session_state.page_state = {'processed': False, 'segments_displayed': False}
            
            # Update previous parameters after clearing state
            st.session_state.previous_params = {
                'youtube_url': youtube_url,
                'content_type': content_type,
                'duration': duration,
                'custom_prompt': custom_prompt
            }
            
            # Force cache clearing
            st.cache_data.clear()
            st.cache_resource.clear()
            
        # Create centered container for loading animation
        loading_container = st.container()
        with loading_container:
            st.markdown(
                """
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 400px;">
                    <div class="loader"></div>
                    <h2 style="margin-top: 2rem; text-align: center;">Generating Your Clips ✨</h2>
                    <p style="color: #666; text-align: center;">This might take a minute...</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Add CSS for loading animation
            st.markdown(
                """
                <style>
                    .loader {
                        width: 150px;
                        height: 150px;
                        border: 8px solid #f3f3f3;
                        border-top: 8px solid #3498db;
                        border-radius: 50%;
                        animation: spin 1s linear infinite;
                    }
                    
                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                </style>
                """,
                unsafe_allow_html=True
            )

            # Process content
            transcript = get_youtube_transcript(youtube_url)
            if transcript:
                st.session_state.transcript = transcript
                st.session_state.current_video_url = youtube_url
                
                analysis_result = analyze_transcript_with_aria(
                    transcript,
                    content_type,
                    duration,
                    custom_prompt
                )
                
                if analysis_result:
                    segments = parse_segments_response(analysis_result)
                    if segments:
                        st.session_state.processed_segments = segments
                        st.session_state.page_state['processed'] = True
                        st.rerun()

    # Show processed segments if they exist
    if st.session_state.processed_segments:
        with tempfile.TemporaryDirectory() as temp_dir:
            for idx, segment in enumerate(st.session_state.processed_segments):
                process_and_display_segment(segment, idx, st.session_state.current_video_url, temp_dir)
    
# Show landing page if no processing has happened
    else:
        # Title and subtitle
        st.markdown("<h1 style='text-align: center; font-size: 3.5rem; margin-bottom: 1rem;'>YouClipAI ✨</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; font-size: 1.8rem; color: #666; margin-bottom: 3rem;'>Transform Your Long Videos into Engaging Short Content</h2>", unsafe_allow_html=True)
        
        # Feature cards using columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                <div style='background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); text-align: center;'>
                    <div style='font-size: 2.5rem; margin-bottom: 1rem;'>🎬</div>
                    <h3 style='font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem;'>Smart Clipping</h3>
                    <p style='color: #666;'>AI-powered identification of the most engaging moments</p>
                </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
                <div style='background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); text-align: center;'>
                    <div style='font-size: 2.5rem; margin-bottom: 1rem;'>✨</div>
                    <h3 style='font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem;'>Auto Captions</h3>
                    <p style='color: #666;'>Generate platform-optimized captions instantly</p>
                </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
                <div style='background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); text-align: center;'>
                    <div style='font-size: 2.5rem; margin-bottom: 1rem;'>⚡</div>
                    <h3 style='font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem;'>Quick Export</h3>
                    <p style='color: #666;'>Download clips ready for social media</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Get started text
        st.markdown("<div style='text-align: center; margin-top: 3rem; color: #666; font-size: 1.1rem;'>👈 Get started by entering a YouTube URL in the sidebar</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()