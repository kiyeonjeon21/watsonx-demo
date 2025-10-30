import requests
from bs4 import BeautifulSoup
from ibm_watsonx_orchestrate.agent_builder.tools import tool

@tool
def get_cognos_info() -> str:
    """
    Fetches and extracts text content from the IBM Cognos Analytics webpage using requests.
    This has all the information about what is Cognos Analytics like: Transform your data into actionable insights, High-quality, high-scale business reporting, Interactive data visualizations with AI-enhanced insights.
    Returns the cleaned text, perfect for answers questions regarding Cognos Analytics.

    Args:
        None

    Returns:
        str: A string containing the extracted text content from the webpage,
             or an error message if the request or parsing fails.
    """
    url= "https://www.ibm.com/products/cognos-analytics"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive"
    }

    try:
        # Send GET request
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise exception for bad status codes

        # Parse HTML content
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove scripts, styles, and other non-content elements
        for element in soup(["script", "style", "header", "footer", "nav"]):
            element.decompose()

        # Extract and clean text
        text = soup.get_text(separator=" ", strip=True)
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        cleaned_text = " ".join(chunk for chunk in chunks if chunk)

        return cleaned_text if cleaned_text else "No content extracted."

    except requests.RequestException as e:
        return f"Error fetching webpage: {str(e)}"
    except Exception as e:
        return f"Error processing webpage: {str(e)}"