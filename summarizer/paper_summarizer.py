import os
import logging
import openai
from openai import OpenAI
import time
import re
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PaperSummarizer:
    def __init__(self):
        """Initialize the paper summarizer with OpenAI API"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "sk-your-openai-key-goes-here":
            logger.warning("OPENAI_API_KEY not set properly in environment variables")
        
        try:
            self.client = OpenAI(api_key=api_key)
            logger.info("Initialized paper summarizer")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None
        
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
        
        # Check if OpenAI client is available
        if not self.client or os.getenv("OPENAI_API_KEY") == "sk-your-openai-key-goes-here":
            logger.warning("OpenAI client not available or using placeholder key, using fallback summary")
            return self._generate_fallback_summary(title, abstract)
        
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
                logger.info(f"Generating summary for paper: {title} (Attempt {attempt+1})")
                
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant that explains complex research papers in simple terms."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
                
                summary = response.choices[0].message.content.strip()
                logger.info(f"Successfully generated summary for: {title}")
                return summary
                
            except Exception as e:
                logger.error(f"Error generating summary (Attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to generate summary after {max_retries} attempts")
                    return self._generate_fallback_summary(title, abstract)

    def _extract_key_sentences(self, text, num_sentences=3):
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

    def _generate_fallback_summary(self, title, abstract):
        """
        Generate a fallback summary when the API call fails
        
        Args:
            title (str): Paper title
            abstract (str): Paper abstract
            
        Returns:
            str: A simple summary based on the title and first few sentences of the abstract
        """
        logger.info(f"Generating fallback summary for: {title}")
        
        # Extract main topic from title
        main_topic = title.split(':')[0] if ':' in title else title
        
        # Get key sentences from abstract
        key_sentences = self._extract_key_sentences(abstract)
        
        # Extract potential methods or approaches
        methods = []
        method_keywords = ["using", "based on", "through", "via", "with the help of", "by"]
        for sentence in key_sentences:
            for keyword in method_keywords:
                if keyword in sentence.lower():
                    method_part = sentence.lower().split(keyword)[1].strip()
                    if method_part and len(method_part) > 5:
                        methods.append(keyword + " " + method_part)
        
        methods_text = methods[0] if methods else "novel techniques described in the paper"
        
        # Create a basic structured summary
        fallback_summary = f"""
[AUTOMATIC FALLBACK SUMMARY - OpenAI API not available]

1. Key Innovation: This paper introduces {main_topic}.

2. Main Finding: Based on the abstract, the research presents advancements in {main_topic.lower()}.

3. Why It Matters: This research contributes to the field of machine learning and may lead to improved understanding or applications in this area.

4. How It Works: The approach works {methods_text}. 

Key points from the abstract:
- {key_sentences[0] if key_sentences else ""}
{f"- {key_sentences[1]}" if len(key_sentences) > 1 else ""}
{f"- {key_sentences[2]}" if len(key_sentences) > 2 else ""}

Note: This is an automatically generated fallback summary because the OpenAI API is unavailable or has reached quota limits.
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
        logger.info(f"Batch summarizing {len(papers)} papers")
        
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