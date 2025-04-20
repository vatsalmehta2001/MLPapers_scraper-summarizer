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
    def __init__(self, force_provider: Optional[str] = None):
        """Initialize the paper summarizer with either OpenAI or Claude API"""
        # Initialize clients
        self.openai_client = None
        self.claude_client = None
        self.api_provider = "None"  # Default if no API is available
        
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
        
    def summarize_paper(self, paper, max_retries=3):
        """
        Generate a layman-friendly summary for the paper
        
        Args:
            paper (dict): Paper data including title, abstract, etc.
            max_retries (int): Maximum number of retries if API call fails
            
        Returns:
            str: A concise summary of the paper
        """
        title = paper['title']
        abstract = paper['abstract']
        
        # If no API client is available, use fallback
        if self.api_provider == "None":
            logger.warning("No API client available, using fallback summary")
            return self._generate_fallback_summary(paper)
        
        prompt = f"""
        Summarize the following machine learning research paper in a concise, easy-to-understand way.
        Make sure the summary is accessible to someone without a deep technical background in machine learning.
        Include the key innovations, results, and potential real-world applications.
        
        Paper Title: {title}
        
        Abstract:
        {abstract}
        
        Create a summary with these sections:
        1. Key Innovation (1-2 sentences)
        2. Main Finding (1-2 sentences)
        3. Why It Matters (1-2 sentences)
        4. How It Works (2-3 sentences in simple terms)
        """
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Generating summary for paper: {title} using {self.api_provider} (Attempt {attempt+1})")
                
                if self.api_provider == "OpenAI" and self.openai_client:
                    summary = self._generate_with_openai(prompt, title)
                elif self.api_provider == "Claude" and self.claude_client:
                    summary = self._generate_with_claude(prompt, title)
                
                if summary:
                    return summary
                                
            except Exception as e:
                logger.error(f"Error generating summary (Attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to generate summary after {max_retries} attempts")
                    return self._generate_fallback_summary(paper)
        
        return self._generate_fallback_summary(paper)
    
    def _generate_with_openai(self, prompt, title):
        """Generate summary using OpenAI API"""
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that explains complex research papers in simple terms."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        summary = response.choices[0].message.content.strip()
        logger.info(f"Successfully generated summary for: {title} with OpenAI")
        return summary
    
    def _generate_with_claude(self, prompt, title):
        """Generate summary using Anthropic Claude API"""
        try:
            logger.info(f"Sending request to Claude API for paper: {title}")
            logger.info(f"Prompt length: {len(prompt)} characters")
            
            # Create a more technical and detailed prompt with bold markdown headings
            enhanced_prompt = f"""
            Please analyze this machine learning research paper in depth from a computer science perspective.
            Focus on technical details, algorithms, and methodologies. Provide a detailed technical summary.

            Paper Title: {title}

            Abstract or Content:
            {prompt}

            Instructions:
            1. Focus on the technical innovations, algorithms, architectures, and methodologies 
            2. Include computational complexity, training details, and technical evaluation metrics
            3. Analyze the code/pseudocode if mentioned, technical datasets used, and implementation details
            4. Discuss technical limitations and potential future CS research directions
            5. Use proper computer science terminology and be precise about technical concepts

            Format your summary with these exact section headings in markdown bold format:

            **SUMMARY**:
            [Provide a technical overview of the paper's contribution to computer science/ML]

            **TECHNICAL CONTRIBUTIONS**:
            - [Technical contribution 1 with CS details]
            - [Technical contribution 2 with CS details]
            - [etc.]

            **METHODOLOGY**:
            [Include specific algorithms, model architectures, complexity analysis, optimization techniques]

            **RESULTS & EVALUATION**:
            [Technical performance metrics, comparison to SOTA, dataset details, statistical significance]

            **SIGNIFICANCE & APPLICATIONS**:
            [Impact on computer science research and potential real-world engineering applications]

            Your response must be technically precise and at a level appropriate for CS/ML practitioners.
            """
            
            response = self.claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,  # Increased for more technical detail
                temperature=0.4,  # Lower for more precise technical responses
                system="You are a computer science expert specializing in machine learning research. Provide technical, detailed summaries focusing on algorithms, methodologies, model architectures, and technical evaluation metrics. Use precise CS terminology and include markdown formatting for structure.",
                messages=[
                    {"role": "user", "content": enhanced_prompt}
                ]
            )
            
            logger.info(f"Claude API response received")
            summary = response.content[0].text.strip()
            
            # Clean up any potential HTML/formatting issues
            summary = summary.replace("<br>", "")
            
            logger.info(f"Successfully generated summary for: {title} with Claude. Summary length: {len(summary)} characters")
            return summary
        except Exception as e:
            logger.error(f"Claude API error with paper '{title}': {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error details: {repr(e)}")
            
            # Try a test call to see if the API is working at all
            try:
                test_response = self.claude_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=10,
                    messages=[{"role": "user", "content": "Hello"}]
                )
                logger.error(f"Test call succeeded, so the issue is with this specific request: {test_response}")
            except Exception as test_error:
                logger.error(f"Test call also failed: {test_error} - API may be completely unavailable")
            
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
        
        # Create a basic structured summary
        fallback_summary = f"""
Key Innovation: This paper introduces research on {main_topic}.

Main Finding: {key_sentences[0] if key_sentences else "The paper presents new findings in this area of research."}

Why It Matters: {impact_text}

How It Works: The approach works {methods_text}. 

Key points from the abstract:
{f"• {key_sentences[0]}" if len(key_sentences) > 0 else ""}
{f"• {key_sentences[1]}" if len(key_sentences) > 1 else ""}
{f"• {key_sentences[2]}" if len(key_sentences) > 2 else ""}
{f"• {key_sentences[3]}" if len(key_sentences) > 3 else ""}
{f"• {key_sentences[4]}" if len(key_sentences) > 4 else ""}

Note: This is an automatically generated summary based on key information extraction.
"""
        return fallback_summary.strip()

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