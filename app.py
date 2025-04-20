import os
import logging
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from dotenv import load_dotenv
from scraper.arxiv_scraper import ArxivScraper
from summarizer.paper_summarizer import PaperSummarizer
from database import DatabaseManager
from sqlalchemy import text
import json
import traceback
import threading
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # for flash messages

# Initialize components
db_path = os.getenv("DATABASE_PATH", "papers.db")
db = DatabaseManager()
arxiv_scraper = ArxivScraper(max_results=int(os.getenv("MAX_PAPERS", "10")))
paper_summarizer = PaperSummarizer()

# Flag to track if scraping is in progress
scraping_in_progress = False

# Schedule-related variables
last_scrape_time = None
scrape_interval = int(os.getenv("SCRAPE_INTERVAL", "24"))  # hours

def background_scrape_and_summarize():
    """Background task to scrape and summarize papers"""
    global scraping_in_progress, last_scrape_time
    
    try:
        scraping_in_progress = True
        logger.info("Starting background scraping and summarization")
        
        # Get papers from arXiv
        categories = os.getenv("CATEGORIES", "cs.LG,cs.AI,cs.CL").split(",")
        days_back = int(os.getenv("DAYS_TO_LOOK_BACK", "7"))
        
        all_papers = []
        for category in categories:
            try:
                logger.info(f"Scraping papers for category: {category}")
                papers = arxiv_scraper.get_recent_papers(days=days_back)
                all_papers.extend(papers)
                logger.info(f"Retrieved {len(papers)} papers for category {category}")
            except Exception as e:
                logger.error(f"Error scraping category {category}: {e}")
        
        logger.info(f"Retrieved a total of {len(all_papers)} papers")
        
        # First store all papers in the database (even without summaries)
        # This ensures we at least see the papers in the UI
        db.add_papers_batch(all_papers)
        
        # Get papers without summaries
        papers_to_summarize = db.get_papers(with_summary_only=False, limit=50)
        # Filter papers without summaries
        papers_to_summarize = [p for p in papers_to_summarize if not p.summary]
        logger.info(f"Found {len(papers_to_summarize)} papers without summaries")
        
        # Summarize papers one by one - ensuring any successful summaries are saved
        if papers_to_summarize:
            logger.info("Starting individual paper summarization")
            
            for paper in papers_to_summarize:
                try:
                    # Convert SQLAlchemy model to dict for the summarizer
                    paper_dict = {
                        'id': paper.arxiv_id,
                        'title': paper.title,
                        'abstract': paper.abstract,
                        'authors': paper.authors_list,
                        'categories': paper.categories_list
                    }
                    
                    # Generate summary for this individual paper
                    summary = paper_summarizer.summarize_paper(paper_dict)
                    
                    # Immediately update paper with summary
                    if summary:
                        db.update_summary(paper.arxiv_id, summary)
                        logger.info(f"Successfully summarized and saved paper: {paper.title}")
                    
                    # Add a small delay to avoid rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error summarizing paper {paper.arxiv_id}: {e}")
                    # Continue with next paper even if this one fails
        
        last_scrape_time = datetime.now()
        logger.info("Completed background scraping and summarization")
        
    except Exception as e:
        logger.error(f"Error in background task: {e}")
        logger.error(traceback.format_exc())
    
    finally:
        scraping_in_progress = False

@app.route('/')
def index():
    """Main page that shows papers with summaries"""
    # Get parameters
    category = request.args.get('category', '')
    search = request.args.get('search', '')
    page = int(request.args.get('page', '1'))
    limit = int(request.args.get('limit', '50'))  # Increased default limit to show more papers
    
    # Get papers based on filters, but don't filter on having summaries
    if search:
        all_papers = db.get_papers(limit=100)
        # Manual search in title and abstract
        papers = [p for p in all_papers if search.lower() in p.title.lower() or 
                 (p.abstract and search.lower() in p.abstract.lower())]
    elif category:
        papers = db.get_papers(category=category, limit=limit)
    else:
        papers = db.get_papers(limit=limit)
    
    # Get all available categories
    all_papers = db.get_papers(limit=100)
    categories = set()
    for paper in all_papers:
        categories.update(paper.categories_list)
    categories = sorted(list(categories))
    
    # Get scraping status info
    next_scrape = None
    if last_scrape_time:
        next_scrape = last_scrape_time + timedelta(hours=scrape_interval)
    
    return render_template('index.html', 
                          papers=papers, 
                          categories=categories,
                          search=search,
                          selected_category=category,
                          scraping_in_progress=scraping_in_progress,
                          last_scrape=last_scrape_time,
                          next_scrape=next_scrape,
                          page=page,
                          limit=limit)

@app.route('/fetch_papers', methods=['POST'])
def fetch_papers():
    """Manually trigger paper fetching and summarization"""
    global scraping_in_progress
    
    if scraping_in_progress:
        flash('Paper fetching is already in progress. Please wait until it completes.', 'warning')
        return redirect(url_for('index'))
    
    # Start the background task in a separate thread
    thread = threading.Thread(target=background_scrape_and_summarize)
    thread.daemon = True
    thread.start()
    
    flash('Started fetching and summarizing papers. This may take a few minutes.', 'info')
    return redirect(url_for('index'))

@app.route('/paper/<arxiv_id>')
def paper_detail(arxiv_id):
    """Show detailed information about a specific paper"""
    paper = db.get_paper_by_id(arxiv_id)
    if not paper:
        flash('Paper not found', 'error')
        return redirect(url_for('index'))
    
    return render_template('paper_detail.html', paper=paper)

@app.route('/admin')
def admin():
    """Admin page for paper management"""
    # Get parameters
    category = request.args.get('category', '')
    search = request.args.get('search', '')
    missing_summary = request.args.get('missing_summary', '') == 'true'
    
    # Get papers based on filters
    if missing_summary:
        papers = db.get_papers(with_summary_only=False, limit=50)
        # Filter in Python for papers without summaries
        papers = [p for p in papers if not p.summary]
    elif search:
        papers = db.get_papers(limit=50)  # No search method, handle in template
    elif category:
        papers = db.get_papers(category=category, limit=50)
    else:
        papers = db.get_papers(limit=50)
    
    # Get all available categories
    all_papers = db.get_papers(limit=100)
    categories = set()
    for paper in all_papers:
        categories.update(paper.categories_list)
    categories = sorted(list(categories))
    
    return render_template('admin.html', 
                          papers=papers, 
                          categories=categories,
                          search=search,
                          selected_category=category,
                          missing_summary=missing_summary)

@app.route('/generate_summary/<arxiv_id>', methods=['POST'])
def generate_summary(arxiv_id):
    """Generate or regenerate a summary for a specific paper"""
    paper = db.get_paper_by_id(arxiv_id)
    if not paper:
        return jsonify({'success': False, 'message': 'Paper not found'})
    
    try:
        # Convert SQLAlchemy model to dict for the summarizer
        paper_dict = {
            'id': paper.arxiv_id,
            'title': paper.title,
            'abstract': paper.abstract,
            'authors': paper.authors_list,
            'categories': paper.categories_list
        }
        
        # Generate summary
        summary = paper_summarizer.summarize_paper(paper_dict)
        
        # Update paper with new summary
        db.update_summary(arxiv_id, summary)
        
        return jsonify({'success': True, 'summary': summary})
    
    except Exception as e:
        logger.error(f"Error generating summary for paper {arxiv_id}: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/delete_paper/<arxiv_id>', methods=['POST'])
def delete_paper(arxiv_id):
    """Delete a paper from the database"""
    # Not implemented in DatabaseManager, would need to add this method
    flash('Delete functionality not implemented', 'error')
    return redirect(url_for('admin'))

@app.route('/debug/api_check')
def debug_api_check():
    """Check the OpenAI API connection status"""
    try:
        # Test if we can initialize the summarizer
        test_summarizer = PaperSummarizer()
        
        # Get API key info (masked for security)
        api_key = os.getenv("OPENAI_API_KEY", "")
        key_status = "Not set"
        if api_key:
            if api_key == "sk-your-openai-key-goes-here":
                key_status = "Using placeholder key"
            else:
                # Mask the API key
                masked_key = f"sk-{'*' * 8}...{api_key[-4:]}" if len(api_key) > 12 else "sk-****"
                key_status = f"Set: {masked_key}"
        
        # Check if client initialized successfully
        client_status = "OK" if test_summarizer.client else "Failed to initialize"
        
        # Try a simple call to test the API
        api_working = False
        error_message = None
        
        if test_summarizer.client:
            try:
                # Make a minimal API call to check if it works
                test_prompt = "Hello, this is a test"
                response = test_summarizer.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": test_prompt}
                    ],
                    max_tokens=5
                )
                api_working = True
            except Exception as e:
                error_message = str(e)
        
        # Return diagnostic information
        return render_template('debug_api.html', 
                              key_status=key_status,
                              client_status=client_status,
                              api_working=api_working,
                              error_message=error_message)
    
    except Exception as e:
        logger.error(f"Error in API check: {e}")
        return render_template('debug_api.html', 
                              error_message=f"Unexpected error: {str(e)}",
                              key_status="Error",
                              client_status="Error",
                              api_working=False)

# Check if database is empty and start initial scrape
# Modern Flask way instead of before_first_request
def check_and_start_scrape():
    papers = db.get_papers(limit=1)
    if not papers:
        logger.info("No papers in database, starting initial scrape")
        thread = threading.Thread(target=background_scrape_and_summarize)
        thread.daemon = True
        thread.start()

# Register function to run after app startup
with app.app_context():
    check_and_start_scrape()

@app.route('/categories')
def categories():
    """Show papers by category"""
    all_papers = db.get_papers(limit=100)
    categories = set()
    for paper in all_papers:
        categories.update(paper.categories_list)
    
    categories = sorted(list(categories))
    return render_template('categories.html', categories=categories)

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

# Add context processor to make year available to all templates
@app.context_processor
def inject_year():
    return {'year': datetime.now().year}

if __name__ == '__main__':
    port = int(os.getenv("PORT", "5001"))
    debug = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1", "t")
    
    logger.info(f"Starting application")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=debug)
    except Exception as e:
        logger.error(f"Error starting Flask app: {e}") 