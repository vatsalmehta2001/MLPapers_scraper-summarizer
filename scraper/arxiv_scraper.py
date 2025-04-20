import arxiv
import os
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ArxivScraper:
    def __init__(self, categories=None, max_results=10):
        """
        Initialize ArxivScraper with categories and max results
        
        Args:
            categories (list): List of arXiv categories to scrape (e.g., ['cs.AI', 'cs.LG'])
            max_results (int): Maximum number of papers to retrieve
        """
        if categories is None:
            categories_str = os.getenv('CATEGORIES', 'cs.AI,cs.LG,cs.CL')
            self.categories = categories_str.split(',')
        else:
            self.categories = categories
            
        self.max_results = int(os.getenv('MAX_PAPERS', max_results))
        logger.info(f"Initialized scraper for categories: {self.categories}, max results: {self.max_results}")
        
    def get_recent_papers(self, days=7):
        """
        Get recent papers from arXiv
        
        Args:
            days (int): How many days back to look for papers
            
        Returns:
            list: List of paper objects with metadata
        """
        logger.info(f"Fetching papers from the last {days} days")
        date_since = datetime.now() - timedelta(days=days)
        
        papers = []
        for category in self.categories:
            query = f"cat:{category} AND submittedDate:[{date_since.strftime('%Y%m%d')}000000 TO now]"
            
            search = arxiv.Search(
                query=query,
                max_results=self.max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            
            try:
                results = list(search.results())
                logger.info(f"Found {len(results)} papers in category {category}")
                papers.extend(results)
            except Exception as e:
                logger.error(f"Error fetching papers for category {category}: {e}")
        
        # Sort papers by published date (newest first)
        papers.sort(key=lambda x: x.published, reverse=True)
        
        # Limit to max_results
        papers = papers[:self.max_results]
        
        return self._format_papers(papers)
    
    def _format_papers(self, papers):
        """
        Format the paper data for easier processing
        
        Args:
            papers (list): List of raw arxiv paper objects
            
        Returns:
            list: List of dictionaries with formatted paper data
        """
        formatted_papers = []
        
        for paper in papers:
            formatted_paper = {
                'id': paper.get_short_id(),
                'title': paper.title,
                'authors': [author.name for author in paper.authors],
                'abstract': paper.summary,
                'pdf_url': paper.pdf_url,
                'arxiv_url': paper.entry_id,
                'published': paper.published.strftime('%Y-%m-%d'),
                'updated': paper.updated.strftime('%Y-%m-%d') if hasattr(paper, 'updated') else None,
                'categories': paper.categories,
                'doi': paper.doi if hasattr(paper, 'doi') else None,
                'journal_ref': paper.journal_ref if hasattr(paper, 'journal_ref') else None,
                'summary': None  # To be filled by the summarizer
            }
            formatted_papers.append(formatted_paper)
            
        return formatted_papers
        
if __name__ == "__main__":
    # Test the scraper
    scraper = ArxivScraper(max_results=5)
    papers = scraper.get_recent_papers(days=3)
    
    for i, paper in enumerate(papers):
        print(f"Paper {i+1}:")
        print(f"Title: {paper['title']}")
        print(f"Authors: {', '.join(paper['authors'])}")
        print(f"Published: {paper['published']}")
        print(f"Categories: {', '.join(paper['categories'])}")
        print(f"URL: {paper['arxiv_url']}")
        print(f"Abstract: {paper['abstract'][:150]}...")
        print("-" * 80) 