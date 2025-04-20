import os
import json
import logging
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create database engine
db_path = os.getenv("DATABASE_PATH", "sqlite:///papers.db")
engine = create_engine(db_path)
Session = sessionmaker(bind=engine)
Base = declarative_base()

class Paper(Base):
    __tablename__ = 'papers'
    
    id = Column(Integer, primary_key=True)
    arxiv_id = Column(String(50), unique=True)
    title = Column(String(255))
    abstract = Column(Text)
    authors = Column(Text)  # Stored as JSON string
    published_date = Column(DateTime)
    updated_date = Column(DateTime, nullable=True)
    categories = Column(Text)  # Stored as JSON string
    arxiv_url = Column(String(255))
    pdf_url = Column(String(255))
    doi = Column(String(100), nullable=True)
    journal_ref = Column(String(255), nullable=True)
    summary = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    
    def __repr__(self):
        return f"<Paper(title='{self.title}', arxiv_id='{self.arxiv_id}')>"
    
    @property
    def authors_list(self):
        """Return authors as a list"""
        if not self.authors:
            return []
        return json.loads(self.authors)
    
    @property
    def categories_list(self):
        """Return categories as a list"""
        if not self.categories:
            return []
        return json.loads(self.categories)

class DatabaseManager:
    def __init__(self):
        """Initialize the database manager"""
        Base.metadata.create_all(engine)
        self.session = Session()
        logger.info(f"Initialized database at {db_path}")
    
    def add_paper(self, paper_data):
        """
        Add a paper to the database
        
        Args:
            paper_data (dict): Paper data dictionary
            
        Returns:
            Paper: The newly created Paper object
        """
        # Check if paper already exists
        existing = self.session.query(Paper).filter_by(arxiv_id=paper_data['id']).first()
        if existing:
            logger.info(f"Paper {paper_data['id']} already exists in database")
            
            # Update the summary if it's provided and the existing one is None
            if paper_data.get('summary') and not existing.summary:
                existing.summary = paper_data['summary']
                self.session.commit()
                logger.info(f"Updated summary for paper {paper_data['id']}")
            # If existing paper has no summary, add the fallback message
            elif not existing.summary:
                existing.summary = "Error: Could not generate summary."
                self.session.commit()
                logger.info(f"Added fallback error message for paper {paper_data['id']}")
                
            return existing
        
        # Parse dates
        published_date = datetime.strptime(paper_data['published'], '%Y-%m-%d')
        updated_date = None
        if paper_data.get('updated'):
            updated_date = datetime.strptime(paper_data['updated'], '%Y-%m-%d')
        
        # Set a default error message if no summary is provided
        summary = paper_data.get('summary')
        if not summary:
            summary = "Error: Could not generate summary."
        
        # Create new paper object
        paper = Paper(
            arxiv_id=paper_data['id'],
            title=paper_data['title'],
            abstract=paper_data['abstract'],
            authors=json.dumps(paper_data['authors']),
            published_date=published_date,
            updated_date=updated_date,
            categories=json.dumps(paper_data['categories']),
            arxiv_url=paper_data['arxiv_url'],
            pdf_url=paper_data['pdf_url'],
            doi=paper_data.get('doi'),
            journal_ref=paper_data.get('journal_ref'),
            summary=summary
        )
        
        self.session.add(paper)
        self.session.commit()
        logger.info(f"Added paper {paper_data['id']} to database")
        return paper
    
    def get_papers(self, limit=10, with_summary_only=False, category=None):
        """
        Get papers from the database
        
        Args:
            limit (int): Maximum number of papers to return
            with_summary_only (bool): If True, only return papers with summaries
            category (str): Filter by category (e.g., 'cs.AI')
            
        Returns:
            list: List of Paper objects
        """
        query = self.session.query(Paper).order_by(Paper.published_date.desc())
        
        if with_summary_only:
            query = query.filter(Paper.summary.isnot(None))
        
        if category:
            # This is a simple filter that checks if the category is in the JSON string
            query = query.filter(Paper.categories.like(f'%{category}%'))
        
        papers = query.limit(limit).all()
        return papers
    
    def get_paper_by_id(self, arxiv_id):
        """
        Get a paper by its arXiv ID
        
        Args:
            arxiv_id (str): arXiv ID
            
        Returns:
            Paper: Paper object or None if not found
        """
        return self.session.query(Paper).filter_by(arxiv_id=arxiv_id).first()
    
    def update_summary(self, arxiv_id, summary):
        """
        Update the summary for a paper
        
        Args:
            arxiv_id (str): arXiv ID
            summary (str): The summary text
            
        Returns:
            bool: True if successful, False otherwise
        """
        paper = self.get_paper_by_id(arxiv_id)
        if not paper:
            logger.error(f"Paper {arxiv_id} not found in database")
            return False
        
        paper.summary = summary
        self.session.commit()
        logger.info(f"Updated summary for paper {arxiv_id}")
        return True
    
    def add_papers_batch(self, papers):
        """
        Add multiple papers to the database
        
        Args:
            papers (list): List of paper dictionaries
            
        Returns:
            list: List of Paper objects
        """
        added_papers = []
        for paper_data in papers:
            paper = self.add_paper(paper_data)
            added_papers.append(paper)
        
        return added_papers
    
    def close(self):
        """Close the database session"""
        self.session.close()
        
if __name__ == "__main__":
    # Test the database
    from scraper.arxiv_scraper import ArxivScraper
    from summarizer.paper_summarizer import PaperSummarizer
    
    db = DatabaseManager()
    
    # Get some papers
    scraper = ArxivScraper(max_results=2)
    papers = scraper.get_recent_papers(days=3)
    
    # Summarize them
    summarizer = PaperSummarizer()
    summarized_papers = summarizer.batch_summarize(papers)
    
    # Add to database
    db.add_papers_batch(summarized_papers)
    
    # Retrieve and print
    stored_papers = db.get_papers(limit=5)
    for paper in stored_papers:
        print(f"Paper: {paper.title}")
        print(f"Summary: {paper.summary}")
        print("-" * 80)
    
    db.close() 