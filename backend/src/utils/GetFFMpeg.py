"""FFmpeg discovery bridge for Go backend migration.

The actual FFmpeg binary discovery logic has been migrated to Go
(internal/utils/getffmpeg.go). This module provides compatibility
shims for code that hasn't been updated yet.
"""

import subprocess
import shutil


def download_ffmpeg(custom_path=None):
    """Compatibility shim - returns first available ffmpeg binary found."""
    # In production, use Go's FindFFMpeg() which checks local bin/, system paths, PATH
    if custom_path and shutil.which(custom_path):
        return custom_path
    
    # Fallback to system ffmpeg
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    
    raise FileNotFoundError("FFmpeg not found in any standard location")


def get_version():
    """Returns FFmpeg version string via subprocess call."""
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        return result.stdout.split('\n')[0] if result.returncode == 0 else ""
    except FileNotFoundError:
        return ""
