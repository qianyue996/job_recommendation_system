import os
import subprocess

from playwright.sync_api import sync_playwright


if __name__ == "__main__":
    chrome_port = 9999

    # # open the local Chrome
    # chrome_path = r'"C:\Program Files\Google\Chrome\Application\chrome.exe"'
    # debugging_port = f"--remote-debugging-port={chrome_port}"
    # command = f"{chrome_path} {debugging_port}"
    # subprocess.Popen(command, shell=True)

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(f"http://localhost:{chrome_port}")
        context = browser.contexts[0]
        page = context.new_page()
        pass