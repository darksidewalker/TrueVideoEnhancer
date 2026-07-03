"""Video info extraction bridge for Go backend migration.

The actual ffprobe-based video metadata extraction has been migrated to Go
(internal/utils/videoinfo.go). This module provides compatibility shims
for code that hasn't been updated yet.
"""

import subprocess
import json
from dataclasses import dataclass


@dataclass
class VideoInfo:
    """Compatibility wrapper for Go's VideoInfo struct."""
    input_path: str
    duration: float = 0.0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    codec: str = ""
    format: str = ""
    pixel_format: str = ""
    nb_frames: int = 0
    has_audio: bool = False
    audio_codec: str = ""
    audio_sample_rate: int = 0


def OpenCVInfo(input_path, ffmpeg_path="ffmpeg"):
    """Extract video metadata using ffprobe (compatibility wrapper)."""
    try:
        cmd = [ffmpeg_path, "-v", "quiet", "-print_format", "json", 
               "-show_format", "-show_streams", input_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return VideoInfo(input_path=input_path)
            
        data = json.loads(result.stdout)
        
        # Extract video stream info
        video_stream = next((s for s in data.get('streams', []) if s['codec_type'] == 'video'), None)
        audio_stream = next((s for s in data.get('streams', []) if s['codec_type'] == 'audio'), None)
        
        if not video_stream:
            return VideoInfo(input_path=input_path)
            
        # Parse frame rate
        fps = 0.0
        if 'r_frame_rate' in video_stream:
            parts = video_stream['r_frame_rate'].split('/')
            if len(parts) == 2:
                try:
                    fps = float(parts[0]) / float(parts[1])
                except:
                    pass
                    
        info = VideoInfo(
            input_path=input_path,
            duration=float(data.get('format', {}).get('duration', 0)),
            fps=fps,
            width=int(video_stream.get('width', 0)),
            height=int(video_stream.get('height', 0)),
            codec=video_stream.get('codec_name', ''),
            format=data.get('format', {}).get('format_name', ''),
            pixel_format=video_stream.get('pix_fmt', ''),
            nb_frames=int(video_stream.get('nb_frames', 0))
        )
        
        if audio_stream:
            info.has_audio = True
            info.audio_codec = audio_stream.get('codec_name', '')
            info.audio_sample_rate = int(audio_stream.get('sample_rate', 0))
            
        return info
        
    except Exception as e:
        return VideoInfo(input_path=input_path)


def print_video_info(info):
    """Print video information in human-readable format."""
    print(f"Input file: {info.input_path}")
    print(f"Duration: {info.duration:.2f}s")
    print(f"Resolution: {info.width}x{info.height}")
    print(f"FPS: {info.fps:.2f}")
    print(f"Codec: {info.codec}")
    print(f"Format: {info.format}")
    print(f"Pixel format: {info.pixel_format}")
    print(f"Frames: {info.nb_frames}")
    
    if info.has_audio:
        print(f"Audio: {info.audio_codec} @ {info.audio_sample_rate}Hz")
    else:
        print("Audio: none")
