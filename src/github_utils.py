
import requests
from pathlib import Path

def get_github_pdf_urls(repo_url):
    """Fetches a list of PDF file URLs from a public GitHub repository directory."""
    # Example repo_url: https://github.com/HR-Chatbot/hr-policy-chatbot/tree/main/policies
    # Convert to API URL: https://api.github.com/repos/HR-Chatbot/hr-policy-chatbot/contents/policies
    api_url = repo_url.replace("https://github.com/", "https://api.github.com/repos/")
    api_url = api_url.replace("/tree/main/", "/contents/")

    try:
        response = requests.get(api_url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        contents = response.json()

        pdf_urls = []
        for item in contents:
            if item["type"] == "file" and item["name"].endswith(".pdf"):
                # Get the raw download URL for the PDF
                pdf_urls.append(item["download_url"])
        return pdf_urls
    except requests.exceptions.RequestException as e:
        print(f"Error fetching GitHub content: {e}")
        return []

def download_pdf(url, save_path):
    """Downloads a PDF file from a given URL to a specified path."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading PDF from {url}: {e}")
        return False
