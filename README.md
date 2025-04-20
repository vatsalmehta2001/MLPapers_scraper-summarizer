# ML Papers Scraper & Summarizer

A personal portfolio project that automatically scrapes recent Machine Learning research papers from arXiv and generates concise, easy-to-understand summaries. Built to enhance my skills in Python, Flask, API integration, and ML content processing.

## Project Purpose

I built this application to:
- Showcase my ability to create full-stack web applications with modern tools
- Demonstrate API integration skills (arXiv API for papers, OpenAI API for summaries)
- Create a practical tool for keeping up with ML research for my own studies
- Practice database design and SQLAlchemy integration

## Features

- Scrapes recent ML papers from arXiv based on selected categories (CS.AI, CS.LG, CS.CL)
- Generates layman-friendly summaries using OpenAI API (with fallback to rule-based summaries)
- Web interface to browse, search and organize papers
- Admin section to manage papers and regenerate summaries
- Responsive Bootstrap UI with dark/light theme

## Notes on OpenAI Integration

The app can work in two modes:
1. **With valid OpenAI API key**: Generates high-quality summaries using GPT models
2. **Without API key/quota**: Falls back to extractive summarization using key sentence extraction

To use the OpenAI integration, you'll need to:
- Create an account at [platform.openai.com](https://platform.openai.com)
- Generate an API key and add billing information
- Add your key to the `.env` file

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   ```
   cp .env.example .env
   # Edit .env with your own API keys
   ```
4. Run the application:
   ```
   python app.py
   ```

## Project Structure

- `scraper/`: Code for scraping research papers from arXiv
- `summarizer/`: NLP models and OpenAI integration for summarization
- `app.py`: Main Flask application with routes and controllers
- `database.py`: SQLAlchemy models and database operations
- `templates/`: Web UI templates using Bootstrap
- `static/`: CSS, JS, and other static assets

## Future Enhancements

- User authentication for personalized paper collections
- PDF parsing for full paper analysis
- Citation network visualization
- Email notifications for new papers in selected topics
- Enhanced offline summarization using local LLMs 