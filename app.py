import os
import sys
import time
import json
import threading
import traceback
import logging
from datetime import datetime, timedelta
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from dotenv import load_dotenv
from sqlalchemy import text
import markdown

# Import our modules
from database import DatabaseManager
from scraper.arxiv_scraper import ArxivScraper
from summarizer.paper_summarizer import PaperSummarizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev_key_change_in_production")

# Add markdown filter to convert markdown to HTML
@app.template_filter('markdown')
def convert_markdown(text):
    return markdown.markdown(text, extensions=['tables', 'fenced_code', 'codehilite'])

# Initialize database manager
db = DatabaseManager()

# Initialize summarizer with PDF extraction enabled
paper_summarizer = PaperSummarizer(pdf_extraction=True)

# Initialize default scraper
arxiv_scraper = ArxivScraper(max_results=int(os.getenv("MAX_PAPERS", "10")))

# Global variables to track scraping status
scraping_in_progress = False
last_scrape_time = None

# Scraping interval in hours
scrape_interval = int(os.getenv("SCRAPE_INTERVAL_HOURS", "12"))

def background_scrape_and_summarize(days_back=None, scraper=None, keywords=None):
    """Background task to scrape and summarize papers"""
    global scraping_in_progress, last_scrape_time
    
    try:
        scraping_in_progress = True
        logger.info("Starting background scraping and summarization")
        
        # Use provided parameters or defaults
        if days_back is None:
            days_back = int(os.getenv("DAYS_TO_LOOK_BACK", "7"))
            
        if scraper is None:
            # Get papers from arXiv
            categories = os.getenv("CATEGORIES", "cs.LG,cs.AI,cs.CL").split(",")
            max_papers = int(os.getenv("MAX_PAPERS", "10"))
            scraper = arxiv_scraper
        
        logger.info(f"Fetching papers from last {days_back} days with keywords: {keywords if keywords else 'None'}")
        
        # Get papers with optional keywords
        all_papers = scraper.get_recent_papers(days=days_back, keywords=keywords)
        
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
    
    # Get parameters from form
    days_back = int(request.form.get('days', os.getenv("DAYS_TO_LOOK_BACK", "3")))
    max_papers = int(request.form.get('max_results', os.getenv("MAX_PAPERS", "5")))
    categories_str = request.form.get('categories', os.getenv("CATEGORIES", "cs.AI,cs.LG,cs.CL"))
    keywords_str = request.form.get('keywords', "")
    
    categories = [c.strip() for c in categories_str.split(',') if c.strip()]
    keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]
    
    # Validate parameters
    if days_back < 1:
        days_back = 1
    elif days_back > 30:
        days_back = 30
        
    if max_papers < 1:
        max_papers = 5
    elif max_papers > 50:
        max_papers = 50
    
    # Initialize scraper with custom params
    custom_scraper = ArxivScraper(categories=categories, max_results=max_papers)
    
    # Start the background task in a separate thread
    thread = threading.Thread(
        target=background_scrape_and_summarize,
        kwargs={'days_back': days_back, 'scraper': custom_scraper, 'keywords': keywords}
    )
    thread.daemon = True
    thread.start()
    
    flash(f'Started fetching and summarizing papers from the last {days_back} days. This may take a few minutes.', 'info')
    return redirect(url_for('index'))

@app.route('/search_papers', methods=['GET', 'POST'])
def search_papers():
    """Search for papers using custom queries"""
    query = request.args.get('query', '') if request.method == 'GET' else request.form.get('query', '')
    
    if not query:
        return render_template('search.html', papers=[], search_query='')
    
    # Initialize scraper
    scraper = ArxivScraper(max_results=20)
    
    # Search for papers
    papers = scraper.search_papers(query)
    
    # Store papers in the database
    if papers:
        db.add_papers_batch(papers)
    
    return render_template('search.html', papers=papers, search_query=query)

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
        papers_without_summaries = [p for p in papers if not p.summary]
    elif search:
        papers = db.get_papers(limit=50)  # No search method, handle in template
        papers_without_summaries = [p for p in papers if not p.summary]
    elif category:
        papers = db.get_papers(category=category, limit=50)
        papers_without_summaries = [p for p in papers if not p.summary]
    else:
        papers = db.get_papers(limit=50)
        papers_without_summaries = [p for p in papers if not p.summary]
    
    # Get all available categories
    all_papers = db.get_papers(limit=100)
    categories = set()
    for paper in all_papers:
        categories.update(paper.categories_list)
    categories = sorted(list(categories))
    
    # Get current API provider
    current_api_provider = paper_summarizer.api_provider
    
    return render_template('admin.html', 
                          papers=papers, 
                          papers_without_summaries=papers_without_summaries,
                          categories=categories,
                          search=search,
                          selected_category=category,
                          missing_summary=missing_summary,
                          current_api_provider=current_api_provider)

@app.route('/set_api_provider', methods=['POST'])
def set_api_provider():
    """Set the preferred API provider for summarization"""
    api_provider = request.form.get('api_provider', 'auto')
    
    try:
        # Reset the paper_summarizer with the selected provider
        global paper_summarizer
        
        if api_provider == 'OpenAI':
            # Force using OpenAI
            paper_summarizer = PaperSummarizer(force_provider='OpenAI', pdf_extraction=True)
        elif api_provider == 'Claude':
            # Force using Claude
            paper_summarizer = PaperSummarizer(force_provider='Claude', pdf_extraction=True)
        else:
            # Auto (use available API)
            paper_summarizer = PaperSummarizer(pdf_extraction=True)
        
        logger.info(f"API provider changed to: {paper_summarizer.api_provider}")
        flash(f'Successfully set API provider to {paper_summarizer.api_provider}', 'success')
    except Exception as e:
        logger.error(f"Error setting API provider: {str(e)}")
        flash(f'Error setting API provider: {str(e)}', 'danger')
    
    return redirect(url_for('admin'))

@app.route('/generate_summary/<arxiv_id>', methods=['POST'])
def generate_summary(arxiv_id):
    """Generate or regenerate a summary for a specific paper"""
    paper = db.get_paper_by_id(arxiv_id)
    if not paper:
        return jsonify({'success': False, 'message': 'Paper not found'})
    
    try:
        # Start timing for performance tracking
        start_time = time.time()
        
        # Convert SQLAlchemy model to dict for the summarizer
        paper_dict = {
            'id': paper.arxiv_id,
            'title': paper.title,
            'abstract': paper.abstract,
            'authors': paper.authors_list,
            'categories': paper.categories_list,
            'pdf_url': paper.pdf_url
        }
        
        logger.info(f"Starting summary generation for paper: {paper.title}")
        logger.info(f"Using API provider: {paper_summarizer.api_provider}")
        logger.info(f"Paper details - ID: {paper.arxiv_id}, Title length: {len(paper.title)}, Abstract length: {len(paper.abstract)}")
        
        # Generate summary
        summary = paper_summarizer.summarize_paper(paper_dict)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Add a timestamp and processing info to make it clear this is a new summary
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        pdf_note = "with full PDF text analysis" if paper_summarizer.pdf_extraction else "from abstract only"
        summary = f"Generated on {timestamp} using {paper_summarizer.api_provider} API {pdf_note} (processed in {processing_time:.1f}s)\n\n{summary}"
        
        # Update paper with new summary
        db.update_summary(arxiv_id, summary)
        logger.info(f"Successfully generated and stored summary for: {paper.title}")
        
        # Redirect to paper detail page to show the new summary
        flash('Summary successfully generated!', 'success')
        return redirect(url_for('paper_detail', arxiv_id=arxiv_id))
    
    except Exception as e:
        logger.error(f"Error generating summary for paper {arxiv_id}: {str(e)}")
        logger.error(traceback.format_exc())  # Log the full stack trace
        flash(f'Error generating summary: {str(e)}', 'danger')
        return redirect(url_for('paper_detail', arxiv_id=arxiv_id))

@app.route('/delete_paper/<arxiv_id>', methods=['POST'])
def delete_paper(arxiv_id):
    """Delete a paper from the database"""
    # Not implemented in DatabaseManager, would need to add this method
    flash('Delete functionality not implemented', 'error')
    return redirect(url_for('admin'))

@app.route('/debug/api_check')
def debug_api_check():
    """Check the API connection status for both OpenAI and Claude"""
    try:
        # Test if we can initialize the summarizer
        test_summarizer = PaperSummarizer()
        
        # Get OpenAI API key info (masked for security)
        openai_api_key = os.getenv("OPENAI_API_KEY", "")
        openai_key_status = "Not set"
        if openai_api_key:
            if openai_api_key == "sk-your-openai-key-goes-here":
                openai_key_status = "Using placeholder key"
            else:
                # Mask the API key
                masked_key = f"sk-{'*' * 8}...{openai_api_key[-4:]}" if len(openai_api_key) > 12 else "sk-****"
                openai_key_status = f"Set: {masked_key}"
        
        # Get Claude API key info (masked for security)
        claude_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        claude_key_status = "Not set"
        if claude_api_key:
            if claude_api_key == "sk-ant-your-key-goes-here":
                claude_key_status = "Using placeholder key"
            else:
                # Mask the API key
                masked_key = f"sk-ant-{'*' * 8}...{claude_api_key[-4:]}" if len(claude_api_key) > 16 else "sk-ant-****"
                claude_key_status = f"Set: {masked_key}"
        
        # Check if clients initialized successfully
        openai_client_status = "OK" if test_summarizer.openai_client else "Failed to initialize"
        claude_client_status = "OK" if test_summarizer.claude_client else "Failed to initialize"
        
        # Try a simple call with the active provider to test
        api_provider = test_summarizer.api_provider
        openai_api_working = False
        claude_api_working = False
        error_message = None
        
        # Test OpenAI if available
        if test_summarizer.openai_client:
            try:
                # Make a minimal API call to check if it works
                test_prompt = "Hello, this is a test"
                response = test_summarizer.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": test_prompt}
                    ],
                    max_tokens=5
                )
                openai_api_working = True
            except Exception as e:
                openai_error_message = str(e)
                if not error_message:
                    error_message = f"OpenAI API Error: {openai_error_message}"
        
        # Test Claude if available
        if test_summarizer.claude_client:
            try:
                # Make a minimal API call to check if it works
                test_prompt = "Hello, this is a test"
                response = test_summarizer.claude_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=5,
                    messages=[
                        {"role": "user", "content": test_prompt}
                    ]
                )
                claude_api_working = True
            except Exception as e:
                claude_error_message = str(e)
                if not error_message:
                    error_message = f"Claude API Error: {claude_error_message}"
        
        # Return diagnostic information
        return render_template('debug_api.html', 
                              openai_key_status=openai_key_status,
                              openai_client_status=openai_client_status,
                              openai_api_working=openai_api_working,
                              claude_key_status=claude_key_status,
                              claude_client_status=claude_client_status,
                              claude_api_working=claude_api_working,
                              api_provider=api_provider,
                              error_message=error_message)
    
    except Exception as e:
        logger.error(f"Error in API check: {e}")
        return render_template('debug_api.html', 
                              error_message=f"Unexpected error: {str(e)}",
                              openai_key_status="Error",
                              openai_client_status="Error",
                              openai_api_working=False,
                              claude_key_status="Error",
                              claude_client_status="Error",
                              claude_api_working=False,
                              api_provider=None)

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

@app.route('/debug/test_claude')
def test_claude_api():
    """Test the Claude API directly"""
    try:
        # Create a test message to send to Claude
        test_message = "Hello, Claude! Can you generate a simple summary for a test question about machine learning?"
        
        logger.info("Attempting to test Claude API with a simple message")
        
        # Create a client
        claude_api_key = os.getenv("ANTHROPIC_API_KEY")
        from anthropic import Anthropic
        client = Anthropic(api_key=claude_api_key)
        
        # Make a test API call
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=50,
            messages=[
                {"role": "user", "content": test_message}
            ]
        )
        
        # If successful, return the response
        logger.info(f"Claude API test successful: {response}")
        
        return jsonify({
            'success': True,
            'message': 'Claude API test successful',
            'response': response.content[0].text
        })
        
    except Exception as e:
        logger.error(f"Error testing Claude API: {str(e)}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'message': f'Claude API test failed: {str(e)}',
            'error': traceback.format_exc()
        })

if __name__ == '__main__':
    port = int(os.getenv("PORT", "5001"))
    debug = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1", "t")
    
    logger.info(f"Starting application")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=debug)
    except Exception as e:
        logger.error(f"Error starting Flask app: {e}") 