import arxiv
import os
from datetime import datetime, timedelta
import logging
import re
import random

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
        
        # List of top ML venues to prioritize
        self.top_venues = [
            'CVPR', 'ICCV', 'ECCV',  # Computer Vision
            'NeurIPS', 'ICML', 'ICLR',  # ML conferences
            'ACL', 'EMNLP', 'NAACL',  # NLP
            'KDD', 'AAAI', 'IJCAI',  # AI/Data Mining
            'TPAMI', 'JMLR', 'TACL',  # Journals
            'arXiv'  # Include arXiv for preprints
        ]
        
    def get_recent_papers(self, days=7, keywords=None):
        """
        Get recent papers from arXiv
        
        Args:
            days (int): How many days back to look for papers
            keywords (list): Optional list of keywords to filter papers
            
        Returns:
            list: List of paper objects with metadata
        """
        logger.info(f"Fetching papers from the last {days} days")
        date_since = datetime.now() - timedelta(days=days)
        
        # Prepare keyword filter if provided
        keyword_filter = ''
        if keywords and len(keywords) > 0:
            keyword_parts = []
            for keyword in keywords:
                # Clean keyword and escape special characters
                clean_keyword = re.sub(r'[^\w\s]', '', keyword).strip()
                if clean_keyword:
                    keyword_parts.append(f'"{clean_keyword}"')
            
            if keyword_parts:
                keyword_filter = ' AND (' + ' OR '.join(keyword_parts) + ')'
        
        papers = []
        for category in self.categories:
            query = f"cat:{category}{keyword_filter} AND submittedDate:[{date_since.strftime('%Y%m%d')}000000 TO now]"
            logger.info(f"Using query: {query}")
            
            search = arxiv.Search(
                query=query,
                max_results=self.max_results * 3,  # Request more to account for filtering
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            
            try:
                results = list(search.results())
                logger.info(f"Found {len(results)} papers in category {category}")
                papers.extend(results)
            except Exception as e:
                logger.error(f"Error fetching papers for category {category}: {e}")
        
        # Remove duplicates (papers can appear in multiple categories)
        unique_papers = {}
        for paper in papers:
            paper_id = paper.get_short_id()
            if paper_id not in unique_papers:
                unique_papers[paper_id] = paper
        
        # Convert back to list
        papers = list(unique_papers.values())
        
        # Sort papers by published date (newest first)
        papers.sort(key=lambda x: x.published, reverse=True)
        
        # Apply quality filtering and ranking
        ranked_papers = self._rank_papers(papers)
        
        # Limit to max_results
        ranked_papers = ranked_papers[:self.max_results]
        
        return self._format_papers(ranked_papers)
    
    def search_papers(self, query, max_results=None):
        """
        Search for papers on arXiv with a specific query
        
        Args:
            query (str): Search query string
            max_results (int): Maximum number of results to return
            
        Returns:
            list: List of paper objects matching the query
        """
        if max_results is None:
            max_results = self.max_results
            
        logger.info(f"Searching for papers with query: {query}")
        
        # Construct search query
        search_terms = query.split()
        query_parts = []
        
        # Add quoted version for exact match on the whole query
        query_parts.append(f'"{query}"')
        
        # Add individual terms
        for term in search_terms:
            if len(term) > 2:  # Skip short terms
                query_parts.append(term)
        
        # Construct arXiv query string
        arxiv_query = ' OR '.join(query_parts)
        if self.categories:
            category_filter = ' OR '.join([f'cat:{cat}' for cat in self.categories])
            arxiv_query = f"({arxiv_query}) AND ({category_filter})"
        
        logger.info(f"Using arXiv query: {arxiv_query}")
        
        # Execute search
        search = arxiv.Search(
            query=arxiv_query,
            max_results=max_results * 2,  # Get more results to account for filtering
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        try:
            results = list(search.results())
            logger.info(f"Found {len(results)} papers matching query")
            
            # Apply ranking to search results
            ranked_results = self._rank_papers(results)
            ranked_results = ranked_results[:max_results]
            
            return self._format_papers(ranked_results)
            
        except Exception as e:
            logger.error(f"Error searching for papers: {e}")
            return []
    
    def _rank_papers(self, papers):
        """
        Rank papers by relevance, venue, citation potential
        
        Args:
            papers (list): List of arxiv paper objects
            
        Returns:
            list: Ranked list of papers
        """
        scored_papers = []
        
        for paper in papers:
            score = 0
            
            # Score based on source/venue (prefer top conferences/journals)
            if hasattr(paper, 'journal_ref') and paper.journal_ref:
                for venue in self.top_venues:
                    if venue.upper() in paper.journal_ref.upper():
                        score += 10
                        break
            
            # Favor papers with more authors (potentially more credible)
            score += min(len(paper.authors) * 0.5, 5)
            
            # Favor papers with longer abstracts (more detailed)
            abstract_len = len(paper.summary) if paper.summary else 0
            score += min(abstract_len / 200, 5)  # Up to 5 points for long abstracts
            
            # Prefer papers with clear technical content
            technical_terms = ['algorithm', 'model', 'framework', 'method', 'approach', 
                            'architecture', 'neural', 'deep learning', 'transformer', 
                            'attention', 'dataset', 'benchmark', 'performance']
            
            abstract_lower = paper.summary.lower() if paper.summary else ""
            for term in technical_terms:
                if term in abstract_lower:
                    score += 0.5
            
            # Add small random factor to avoid identical scores
            score += random.uniform(0, 0.1)
            
            scored_papers.append((paper, score))
        
        # Sort by score (descending)
        scored_papers.sort(key=lambda x: x[1], reverse=True)
        
        # Extract papers from scored list
        ranked_papers = [paper for paper, _ in scored_papers]
        
        return ranked_papers
    
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