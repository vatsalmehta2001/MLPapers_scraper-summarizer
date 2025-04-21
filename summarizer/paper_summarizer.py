import os
import logging
import time
import re
import requests
import json
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List
import random
from datetime import datetime
import tempfile
import traceback
import base64

# Import for OpenAI
try:
    from openai import OpenAI
    from openai.types.chat import ChatCompletion
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Import for Anthropic/Claude
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Import for PDF extraction
try:
    import PyPDF2
    import pdfplumber
    PDF_EXTRACTION_AVAILABLE = True
except ImportError:
    PDF_EXTRACTION_AVAILABLE = False

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("summarizer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PaperSummarizer:
    def __init__(self, force_provider: Optional[str] = None, pdf_extraction: bool = True):
        """Initialize the paper summarizer with either OpenAI or Claude API"""
        # Initialize clients
        self.openai_client = None
        self.claude_client = None
        self.api_provider = "None"  # Default if no API is available
        self.pdf_extraction = pdf_extraction and PDF_EXTRACTION_AVAILABLE
        
        # Initialize Claude client if API key is available (prioritize Claude)
        claude_api_key = os.getenv("ANTHROPIC_API_KEY")
        if ANTHROPIC_AVAILABLE and claude_api_key and claude_api_key != "sk-ant-your-key-goes-here":
            try:
                self.claude_client = Anthropic(api_key=claude_api_key)
                logger.info("Claude client initialized successfully")
                # Use Claude by default or if Claude is forced
                if force_provider != "OpenAI":
                    self.api_provider = "Claude"
            except Exception as e:
                logger.error(f"Error initializing Claude client: {e}")
        else:
            logger.warning("Claude API not available (missing key or package)")
        
        # Initialize OpenAI client if API key is available and Claude didn't initialize or OpenAI is forced
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if OPENAI_AVAILABLE and openai_api_key and openai_api_key != "sk-your-openai-key-goes-here":
            try:
                self.openai_client = OpenAI(api_key=openai_api_key)
                logger.info("OpenAI client initialized successfully")
                # Only use OpenAI if Claude not available or OpenAI is forced
                if (self.api_provider == "None" or force_provider == "OpenAI"):
                    self.api_provider = "OpenAI"
            except Exception as e:
                logger.error(f"Error initializing OpenAI client: {e}")
        else:
            logger.warning("OpenAI API not available (missing key or package)")
        
        # Handle forcing a provider that's not available
        if force_provider == "OpenAI" and not self.openai_client:
            logger.error("Forced OpenAI provider but client not available")
            
        if force_provider == "Claude" and not self.claude_client:
            logger.error("Forced Claude provider but client not available")
        
        # Log final selection
        logger.info(f"Using {self.api_provider} API provider for summarization")
        logger.info(f"PDF extraction is {'enabled' if self.pdf_extraction else 'disabled'}")
        
    def download_pdf(self, pdf_url: str) -> Optional[str]:
        """Download a PDF from a URL
        
        Args:
            pdf_url: URL of the PDF to download
            
        Returns:
            Path to the downloaded PDF, or None if download failed
        """
        try:
            # Create a temporary file to store the PDF
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, f"paper_{int(time.time())}.pdf")
            
            # Download the PDF
            logger.info(f"Downloading PDF from {pdf_url}")
            response = requests.get(pdf_url, stream=True, timeout=30)
            if response.status_code != 200:
                logger.error(f"Failed to download PDF: {response.status_code}")
                return None
                
            # Write the PDF to the temporary file
            with open(temp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            logger.info(f"PDF downloaded to {temp_file}")
            return temp_file
        except Exception as e:
            logger.error(f"Error downloading PDF: {e}")
            return None
    
    def extract_text_from_pdf(self, pdf_path: str, max_pages: int = 20) -> str:
        """Extract text from a PDF file
        
        Args:
            pdf_path: Path to the PDF file
            max_pages: Maximum number of pages to extract
            
        Returns:
            Extracted text from the PDF
        """
        if not self.pdf_extraction:
            logger.warning("PDF extraction is disabled")
            return ""
            
        try:
            logger.info(f"Extracting text from PDF: {pdf_path}")
            
            # Try pdfplumber first (better quality but slower)
            try:
                text_parts = []
                with pdfplumber.open(pdf_path) as pdf:
                    # Limit to max_pages to avoid processing very large PDFs
                    for i, page in enumerate(pdf.pages[:max_pages]):
                        text = page.extract_text()
                        if text:
                            text_parts.append(text)
                
                full_text = "\n\n".join(text_parts)
                
                # If we got meaningful text, return it
                if len(full_text) > 100:
                    logger.info(f"Successfully extracted {len(full_text)} characters from PDF using pdfplumber")
                    return full_text
            except Exception as e:
                logger.warning(f"Error using pdfplumber: {e}, falling back to PyPDF2")
                
            # Fall back to PyPDF2
            text_parts = []
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                # Limit to max_pages to avoid processing very large PDFs
                for i in range(min(len(reader.pages), max_pages)):
                    page = reader.pages[i]
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
            
            full_text = "\n\n".join(text_parts)
            logger.info(f"Successfully extracted {len(full_text)} characters from PDF using PyPDF2")
            return full_text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""
    
    def get_paper_full_text(self, paper: Dict[str, Any]) -> str:
        """Get the full text of a paper by downloading and extracting from PDF
        
        Args:
            paper: Dictionary containing paper details
            
        Returns:
            The full text of the paper, or empty string if extraction failed
        """
        if not self.pdf_extraction:
            return ""
            
        try:
            # Get the PDF URL
            pdf_url = paper.get('pdf_url')
            if not pdf_url:
                logger.warning(f"No PDF URL for paper {paper['id']}")
                return ""
                
            # Download the PDF
            pdf_path = self.download_pdf(pdf_url)
            if not pdf_path:
                logger.warning(f"Failed to download PDF for paper {paper['id']}")
                return ""
                
            # Extract text from the PDF
            full_text = self.extract_text_from_pdf(pdf_path)
            
            # Clean up the temporary file
            try:
                os.unlink(pdf_path)
            except:
                pass
                
            return full_text
        except Exception as e:
            logger.error(f"Error getting full text for paper {paper['id']}: {e}")
            return ""
            
    def summarize_paper(self, paper):
        """
        Generate a summary for a research paper.
        
        Args:
            paper (dict): Dictionary with paper details including id, title, abstract, etc.
            
        Returns:
            str: Generated summary or None if generation failed.
        """
        try:
            logger.info(f"Summarizing paper: {paper['id']}")
            
            # Get more content from PDF if available
            paper_content = self.get_paper_full_text(paper) if self.pdf_extraction else None
            
            # Use the full paper content if available, otherwise fallback to abstract
            content_to_summarize = paper_content or paper['abstract']
            
            # Try to generate summary with the current provider
            max_retries = 3
            retry_count = 0
            backoff_time = 2  # Start with 2 seconds
            
            while retry_count < max_retries:
                try:
                    # Log which provider we're using
                    logger.info(f"Attempting to summarize using {self.api_provider}")
                    
                    # Generate summary based on provider
                    if self.api_provider == "Claude" and self.claude_client:
                        summary = self._generate_claude_summary(paper, content_to_summarize)
                        return summary
                    elif self.api_provider == "OpenAI" and self.openai_client:
                        summary = self._generate_openai_summary(paper, content_to_summarize)
                        return summary
                    else:
                        # This should never happen due to initialization, but handle it anyway
                        logger.error(f"No API client available for {self.api_provider}")
                        # Try the other provider if available
                        if self.api_provider == "Claude" and self.openai_client:
                            logger.info("Falling back to OpenAI")
                            self.api_provider = "OpenAI"
                            continue
                        elif self.api_provider == "OpenAI" and self.claude_client:
                            logger.info("Falling back to Claude")
                            self.api_provider = "Claude"
                            continue
                        else:
                            # If no APIs are available, generate fallback summary
                            return self._generate_fallback_summary(paper)
                    
                except Exception as e:
                    retry_count += 1
                    error_type = type(e).__name__
                    logger.warning(f"API error ({error_type}) on attempt {retry_count}/{max_retries}: {str(e)}")
                    
                    # If we have multiple API options, try switching on failure
                    if retry_count >= max_retries // 2:
                        if self.api_provider == "Claude" and self.openai_client:
                            logger.info("Switching to OpenAI after Claude API error")
                            self.api_provider = "OpenAI"
                            retry_count = max_retries // 2  # Reset counter partly to give the new API a fair chance
                            continue
                        elif self.api_provider == "OpenAI" and self.claude_client:
                            logger.info("Switching to Claude after OpenAI API error")
                            self.api_provider = "Claude"
                            retry_count = max_retries // 2  # Reset counter partly
                            continue
                    
                    if retry_count < max_retries:
                        sleep_time = backoff_time * (2 ** (retry_count - 1))  # Exponential backoff
                        logger.info(f"Retrying in {sleep_time} seconds...")
                        time.sleep(sleep_time)
            
            # If we've exhausted all retries and APIs, use fallback
            logger.warning("All API attempts failed, using fallback summary")
            return self._generate_fallback_summary(paper)
            
        except Exception as e:
            logger.error(f"Error in summarize_paper: {e}")
            logger.error(traceback.format_exc())
            # Return a minimal fallback summary when everything else fails
            return self._generate_minimal_fallback(paper)
        
    def _generate_openai_summary(self, paper: Dict[str, Any], full_text: str = "") -> str:
        """Generate a summary using OpenAI API
        
        Args:
            paper: Dictionary containing paper details
            full_text: Full text of the paper extracted from PDF
            
        Returns:
            A structured summary of the paper
        """
        # Create prompt for the API
        system_prompt = """You are a helpful research assistant that specializes in summarizing machine learning research papers.
Given information about a paper, create a comprehensive but concise summary in the following format:

# SUMMARY
[2-3 sentences overview of what the paper is about and its main contribution]

# KEY POINTS
- [Key point 1]
- [Key point 2]
- [Key point 3]
- [Add more points as needed]

# METHODOLOGY
[2-3 sentences describing the approach/methodology]

# RESULTS
[1-2 sentences on the main results and their significance]

# IMPLICATIONS
[1-2 sentences on why this matters and potential applications]

Keep the entire summary under 400 words. Focus on the most important aspects of the research."""
        
        # Build the paper content
        paper_content = f"""Title: {paper['title']}
Authors: {', '.join(paper['authors'])}
Categories: {', '.join(paper['categories'])}
Abstract: {paper['abstract']}"""

        # Add full text if available
        if full_text:
            # Truncate full text to avoid token limits
            max_fulltext_length = 8000
            if len(full_text) > max_fulltext_length:
                truncated_text = full_text[:max_fulltext_length]
                paper_content += f"\n\nFull Paper Text (truncated): {truncated_text}..."
            else:
                paper_content += f"\n\nFull Paper Text: {full_text}"
        
        user_prompt = f"""Please summarize the following research paper:

{paper_content}
"""
        
        # Make the API call
        logger.info(f"Making OpenAI API call to summarize paper: {paper['id']}")
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo-16k",  # Use a model with larger context window
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )
        
        # Extract and return the summary
        summary = response.choices[0].message.content
        logger.info(f"Successfully generated OpenAI summary for paper {paper['id']}")
        return summary

    def _generate_claude_summary(self, paper: Dict[str, Any], full_text: str = "") -> str:
        """Generate a summary using Claude API with enhanced technical detail and formatting
        
        Args:
            paper: Dictionary containing paper details
            full_text: Full text of the paper extracted from PDF
            
        Returns:
            A well-formatted, technical summary of the paper
        """
        try:
            logger.info(f"Sending request to Claude API for paper: {paper['id']}")
            
            # Create a detailed technical prompt with clear formatting instructions
            system_prompt = """You are an expert computer science researcher specializing in analyzing and summarizing ML/AI research papers.
Your task is to create technically precise, comprehensive summaries of computer science/AI papers.
Focus on technical details, algorithmic approaches, model architectures, experimental methodology, and results.
Use formal academic language and technical terminology appropriate for computer science audiences.
Your summaries must be thorough, technically accurate, and include implementation details from the full paper.

Structure your response with proper markdown formatting including code snippets where relevant.
Extract specific metrics, hyperparameters, dataset details, and implementation details from the paper."""
            
            # Build the paper content
            paper_content = f"""## Paper Details:
- **Title**: {paper['title']}
- **Authors**: {', '.join(paper['authors'])}
- **Categories**: {', '.join(paper['categories'])}

## Abstract:
{paper['abstract']}"""

            # Add full text if available, with a clear marker
            if full_text:
                # Truncate full text to avoid token limits
                max_fulltext_length = 60000  # Claude can handle longer contexts
                if len(full_text) > max_fulltext_length:
                    truncated_text = full_text[:max_fulltext_length]
                    paper_content += f"\n\n## Full Paper Text (truncated):\n{truncated_text}..."
                else:
                    paper_content += f"\n\n## Full Paper Text:\n{full_text}"
            
            user_prompt = f"""# Paper Analysis Request

{paper_content}

## Instructions:
Analyze this computer science research paper and create a comprehensive, technically detailed summary.
Your summary should reflect a thorough understanding of the FULL PAPER, not just the abstract.

Your summary MUST use the following markdown structure:

# {paper['title']}

## SUMMARY
[3-4 paragraphs providing a detailed technical overview of the paper's contribution, methodology, and results]

## KEY CONTRIBUTIONS
- [Key technical contribution 1 with specific details]
- [Key technical contribution 2 with specific details]
- [Key technical contribution 3 with specific details]
- [Add more as needed, keeping each point technical and specific]

## METHODOLOGY
[4-6 sentences describing the technical approach in detail, including algorithms, models, frameworks used]

## TECHNICAL DETAILS
- **Architecture**: [Detailed description of model architecture, with parameters, layers, and connectivity]
- **Datasets**: [Comprehensive details about datasets, including sizes, preprocessing, and splits]
- **Training**: [Specific training methodology, optimization algorithms, loss functions, hyperparameters]
- **Evaluation**: [Detailed evaluation metrics, comparison methodologies, and statistical analysis]

## RESULTS
[3-4 sentences on quantitative results with specific numbers and comparisons to baseline/SOTA methods]

## IMPLICATIONS & FUTURE WORK
[2-3 sentences on technical impact and potential future research directions]

Ensure all headings are properly formatted with markdown, text is well-spaced, and the summary is technically precise.
Extract specific technical details from the full paper text whenever possible - include actual numbers, parameters, and technical specifications.
"""
            
            # Make the API call with a technical focus and increased token limit
            model = "claude-3-opus-20240229"  # Use the most capable Claude model
            logger.info(f"Using {model} to generate summary")
            
            response = self.claude_client.messages.create(
                model=model,
                max_tokens=4000,  # Significantly increased token count for more detailed summaries
                temperature=0.2,  # Lower temperature for more precise technical content
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            summary = response.content[0].text.strip()
            
            # Ensure proper formatting is preserved
            summary = summary.replace("\n\n\n", "\n\n").replace("\n\n\n\n", "\n\n")
            
            logger.info(f"Successfully generated Claude summary for paper {paper['id']}")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating Claude summary for paper {paper['id']}: {e}")
            raise

    def _extract_key_sentences(self, text, num_sentences=5):
        """Extract important sentences from text"""
        # Split text into sentences
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # If we have few sentences, return all of them
        if len(sentences) <= num_sentences:
            return sentences
            
        # Otherwise, get first, middle and last sentence as they often contain key information
        first = sentences[0]
        middle = sentences[len(sentences) // 2]
        last = sentences[-1]
        
        # Return unique sentences only
        selected = []
        for s in [first, middle, last]:
            if s not in selected:
                selected.append(s)
                
        return selected[:num_sentences]

    def _generate_fallback_summary(self, paper):
        """
        Generate a fallback summary when the API call fails
        
        Args:
            paper (dict): Paper data including title, abstract, etc.
            
        Returns:
            str: A simple summary based on the title and first few sentences of the abstract
        """
        logger.info(f"Generating fallback summary for: {paper['title']}")
        
        # Extract main topic from title
        main_topic = paper['title'].split(':')[0] if ':' in paper['title'] else paper['title']
        
        # Get key sentences from abstract (increased to 5)
        key_sentences = self._extract_key_sentences(paper['abstract'], num_sentences=5)
        
        # Extract potential methods or approaches
        methods = []
        method_keywords = ["using", "based on", "through", "via", "with the help of", "by", "propose", "introduce", "develop", "present"]
        for sentence in key_sentences:
            for keyword in method_keywords:
                if keyword in sentence.lower():
                    method_part = sentence.lower().split(keyword)[1].strip()
                    if method_part and len(method_part) > 5:
                        methods.append(keyword + " " + method_part)
        
        methods_text = methods[0] if methods else "the approach described in the abstract"
        
        # Try to extract potential impact
        impact_sentences = []
        impact_keywords = ["improve", "enhance", "advance", "better", "outperform", "state-of-the-art", "sota", "novel", "new", "first"]
        for sentence in key_sentences:
            for keyword in impact_keywords:
                if keyword in sentence.lower():
                    impact_sentences.append(sentence)
                    break
        
        impact_text = impact_sentences[0] if impact_sentences else "It contributes to advancing the field of machine learning research."
        
        # Try to extract technical details if they exist in the abstract
        tech_details = {}
        technique_patterns = [
            (r'architecture[s]?\s*[:;]\s*([^.]+)', 'Architecture'),
            (r'model[s]?\s*[:;]\s*([^.]+)', 'Architecture'),
            (r'dataset[s]?\s*[:;]\s*([^.]+)', 'Datasets'),
            (r'data\s*[:;]\s*([^.]+)', 'Datasets'),
            (r'train(?:ing|ed)\s*[:;]\s*([^.]+)', 'Training'),
            (r'learning rate[s]?\s*[:;]\s*([^.]+)', 'Training'),
            (r'batch size[s]?\s*[:;]\s*([^.]+)', 'Training'),
            (r'epochs\s*[:;]\s*([^.]+)', 'Training'),
            (r'evaluation\s*[:;]\s*([^.]+)', 'Evaluation'),
            (r'result[s]?\s*[:;]\s*([^.]+)', 'Results'),
            (r'performance\s*[:;]\s*([^.]+)', 'Results'),
            (r'accuracy\s*[:;]\s*([^.]+)', 'Results'),
        ]
        
        for pattern, category in technique_patterns:
            matches = re.findall(pattern, paper['abstract'], re.IGNORECASE)
            if matches:
                tech_details[category] = tech_details.get(category, []) + matches
        
        # Create a basic structured summary with enhanced formatting
        fallback_summary = f"""
# SUMMARY

Key Innovation: This paper introduces research on {main_topic}.

Main Finding: {key_sentences[0] if key_sentences else "The paper presents new findings in this area of research."}

Why It Matters: {impact_text}

How It Works: The approach works {methods_text}.

## Key points from the abstract:

* {key_sentences[0] if len(key_sentences) > 0 else ""}
* {key_sentences[1] if len(key_sentences) > 1 else ""}
* {key_sentences[2] if len(key_sentences) > 2 else ""}
* {key_sentences[3] if len(key_sentences) > 3 else ""}
* {key_sentences[4] if len(key_sentences) > 4 else ""}

## TECHNICAL DETAILS

* <span class="category-label">Architecture</span> : {tech_details.get('Architecture', ['Vision encoder connected to decoder with MLP projector.'])[0] if 'Architecture' in tech_details else 'Not specified in abstract.'}

* <span class="category-label">Datasets</span> : {tech_details.get('Datasets', ['Combined synthetic and real-world data.'])[0] if 'Datasets' in tech_details else 'Not specified in abstract.'}

* <span class="category-label">Training</span> : {tech_details.get('Training', ['Multi-stage training process.'])[0] if 'Training' in tech_details else 'Training details not provided in abstract.'}

* <span class="category-label">Evaluation</span> : {tech_details.get('Evaluation', ['Standard benchmarks for this research area.'])[0] if 'Evaluation' in tech_details else 'Evaluation methodology not specified in abstract.'}

## RESULTS

{tech_details.get('Results', ['The authors report improved performance compared to previous methods.'])[0] if 'Results' in tech_details else 'Results details not provided in abstract.'}

Note: This is an automatically generated summary based on key information extraction.
"""
        return fallback_summary.strip()

    def _generate_minimal_fallback(self, paper):
        """Generate a very simple fallback summary when all other methods fail"""
        try:
            # Extract 2-3 sentences from the abstract
            key_sentences = self._extract_key_sentences(paper['abstract'], 3)
            
            return f"""
# Summary

**Note: This is an auto-extracted summary due to API limitations.**

{' '.join(key_sentences)}

# Key Points

- This paper titled "{paper['title']}" was written by {', '.join(paper['authors'][:3])}{"..." if len(paper['authors']) > 3 else ""}.
- Published in the following categories: {', '.join(paper['categories'])}.
- For a better summary, please try again later or check the full paper.
"""
        except Exception:
            # Absolute minimal fallback
            return f"Unable to generate summary for '{paper['title']}'. Please check the paper abstract directly."

    def batch_summarize(self, papers):
        """
        Generate summaries for a batch of papers
        
        Args:
            papers (list): List of paper dictionaries
            
        Returns:
            list: The same papers with summaries added
        """
        logger.info(f"Batch summarizing {len(papers)} papers using {self.api_provider if self.api_provider != 'None' else 'fallback'}")
        
        for i, paper in enumerate(papers):
            summary = self.summarize_paper(paper)
            papers[i]['summary'] = summary
            # Add a small delay to avoid rate limiting
            time.sleep(1)
            
        return papers

    def summarize_paper_with_direct_pdf(self, paper):
        """Generate summary by sending the PDF directly to Claude using multimodal capabilities
        
        Args:
            paper: Dictionary with paper details including id, title, abstract, etc.
            
        Returns:
            str: Generated summary or None if generation failed.
        """
        try:
            logger.info(f"Attempting to summarize using direct PDF upload for: {paper['id']}")
            
            # Only proceed if we have Claude client and PDF URL
            if not self.claude_client or not paper.get('pdf_url'):
                logger.warning("Claude client not available or PDF URL missing")
                return self.summarize_paper(paper)
            
            # Download the PDF
            pdf_path = self.download_pdf(paper['pdf_url'])
            if not pdf_path:
                logger.warning(f"Failed to download PDF for {paper['id']}, falling back to standard method")
                return self.summarize_paper(paper)
            
            try:
                # Read the PDF as binary data
                with open(pdf_path, 'rb') as f:
                    pdf_content = f.read()
                    
                # Create system and user messages
                system_prompt = """You are an expert computer science researcher specializing in analyzing technical ML/AI research papers.
Create comprehensive, technically precise summaries focusing on algorithms, architectures, methodologies, and results.
Use proper academic language and technical terminology for a computer science audience.
Extract specific implementation details, metrics, hyperparameters, and technical insights from the full paper."""

                # Create a multipart message with the PDF content
                response = self.claude_client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=4000,
                    temperature=0.2,
                    system=system_prompt,
                    messages=[
                        {
                            "role": "user", 
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"""Please analyze this computer science research paper titled "{paper['title']}" and provide a comprehensive technical summary.

Focus on the technical details, methodology, algorithms, architectures, and experimental results.
Extract specific numbers, methodologies, and implementation details from the paper.

Format your response with these sections:
1. SUMMARY - A detailed technical overview (3-4 paragraphs)
2. KEY CONTRIBUTIONS - Bullet points of specific technical contributions
3. METHODOLOGY - Detailed explanation of methods, algorithms, and approaches
4. TECHNICAL DETAILS - Architecture specifics, datasets, training procedures, evaluation metrics
5. RESULTS - Quantitative results with specific numbers and comparisons
6. IMPLICATIONS & FUTURE WORK

Make this summary technically precise and comprehensive, suitable for computer science researchers."""
                                },
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "application/pdf",
                                        "data": base64.b64encode(pdf_content).decode('utf-8')
                                    }
                                }
                            ]
                        }
                    ]
                )
                
                # Extract the summary from the response
                summary = response.content[0].text.strip()
                
                # Add metadata about generation method
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                summary_with_meta = f"Generated on {timestamp} using Claude API with direct PDF analysis\n\n{summary}"
                
                logger.info(f"Successfully generated summary via direct PDF upload for {paper['id']}")
                
                # Clean up the temporary file
                try:
                    os.unlink(pdf_path)
                except:
                    pass
                
                return summary_with_meta
                
            except Exception as e:
                logger.error(f"Error in direct PDF summarization: {e}")
                logger.error(traceback.format_exc())
                # Clean up the temporary file
                try:
                    os.unlink(pdf_path)
                except:
                    pass
                # Fall back to regular summarization
                return self.summarize_paper(paper)
            
        except Exception as e:
            logger.error(f"Error in direct PDF summarization outer try block: {e}")
            logger.error(traceback.format_exc())
            return self.summarize_paper(paper)

if __name__ == "__main__":
    # Test the summarizer
    from scraper.arxiv_scraper import ArxivScraper
    
    scraper = ArxivScraper(max_results=2)
    papers = scraper.get_recent_papers(days=3)
    
    summarizer = PaperSummarizer()
    summarized_papers = summarizer.batch_summarize(papers)
    
    for i, paper in enumerate(summarized_papers):
        print(f"Paper {i+1}: {paper['title']}")
        print(f"Summary:\n{paper['summary']}")
        print("-" * 80) 