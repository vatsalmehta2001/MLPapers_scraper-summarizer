import os
import time
import logging
import schedule
from datetime import datetime
from dotenv import load_dotenv
from scraper.arxiv_scraper import ArxivScraper
from summarizer.paper_summarizer import PaperSummarizer
from database import DatabaseManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='scheduler.log'
)
logger = logging.getLogger(__name__)

def fetch_and_summarize():
    """Fetch new papers and summarize them"""
    try:
        logger.info("Starting scheduled paper fetch and summarization")
        
        # Get environment variables
        days = int(os.getenv('DAYS_TO_LOOK_BACK', 1))
        max_results = int(os.getenv('MAX_PAPERS', 10))
        categories = os.getenv('CATEGORIES', 'cs.AI,cs.LG,cs.CL').split(',')
        
        # Initialize components
        db = DatabaseManager()
        scraper = ArxivScraper(categories=categories, max_results=max_results)
        summarizer = PaperSummarizer()
        
        # Fetch papers
        logger.info(f"Fetching papers from the last {days} days for categories: {categories}")
        papers = scraper.get_recent_papers(days=days)
        
        if not papers:
            logger.info("No new papers found")
            db.close()
            return
        
        # Summarize papers
        logger.info(f"Summarizing {len(papers)} papers")
        summarized_papers = summarizer.batch_summarize(papers)
        
        # Store in database
        logger.info("Storing papers in database")
        db.add_papers_batch(summarized_papers)
        
        # Close database connection
        db.close()
        
        logger.info(f"Successfully processed {len(summarized_papers)} papers")
        
    except Exception as e:
        logger.error(f"Error in scheduled task: {e}")

def main():
    """Main function to run the scheduler"""
    # Get schedule interval from environment variables
    interval_hours = int(os.getenv('SCRAPE_INTERVAL', 24))
    
    # Run once immediately
    fetch_and_summarize()
    
    # Schedule to run at regular intervals
    schedule.every(interval_hours).hours.do(fetch_and_summarize)
    
    logger.info(f"Scheduler started, will run every {interval_hours} hours")
    
    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    main() 