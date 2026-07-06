#!/usr/bin/env python3
"""Take a screenshot of DaSiWa True Video Enhancer UI"""

from playwright.sync_api import sync_playwright
import os

def take_screenshot():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1200, "height": 900})
        
        # Navigate to the local app
        page.goto("http://localhost:8612")
        
        # Wait for content to load
        page.wait_for_load_state("networkidle")
        
        # Take screenshot
        screenshot_path = "assets/preview.png"
        os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
        page.screenshot(path=screenshot_path, full_page=True)
        
        print(f"Screenshot saved to {screenshot_path}")
        browser.close()

if __name__ == "__main__":
    take_screenshot()