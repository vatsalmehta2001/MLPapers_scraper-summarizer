# MLPapers Scraper and Summarizer

A web application that scrapes recent Machine Learning research papers from arXiv and generates summaries using either OpenAI or Claude API.

## Features

- Automated scraping of recent ML research papers from arXiv
- AI-powered paper summarization with OpenAI or Claude
- Web interface to browse and read papers and summaries
- Admin panel to manage papers and trigger manual updates
- Flexible API provider selection - use either OpenAI or Claude

## Setup

1. Clone the repository
   ```
   git clone <repository-url>
   cd MLPapers_scraper-summarizer
   ```

2. Create and activate a virtual environment
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables
   - Copy `.env.example` to `.env`
   - Configure your API keys (see API Configuration section)

5. Run the application
   ```
   python app.py
   ```
   
6. Access the web interface at `http://localhost:5001`

## API Configuration

The application supports two AI providers for generating summaries:

### Option 1: OpenAI API
1. Go to https://platform.openai.com/account/api-keys to get your API key
2. Add your key to the `.env` file:
   ```
   OPENAI_API_KEY=sk-your-openai-key-goes-here
   ```

### Option 2: Claude API
1. Go to https://console.anthropic.com/ to get your API key
2. Add your key to the `.env` file:
   ```
   ANTHROPIC_API_KEY=sk-ant-your-key-goes-here
   ```

You can provide either one or both API keys. The application is designed to work with just one of the services if needed.

## Switching Between API Providers

In the Admin Panel, you can select which API provider to use for summarization:

1. **Auto (default)**: The application will automatically use available APIs, with Claude preferred if both are available
2. **OpenAI**: Force using OpenAI's GPT model even if Claude is available
3. **Claude**: Force using Claude even if OpenAI is available

You can also check the API connection status with the API Diagnostics tool in the Admin Panel.

## Usage

- **Home Page**: View recently scraped papers
- **Paper Details**: Click on a paper to view its details and AI-generated summary
- **Admin Panel**: Access admin functionality to manage papers and trigger updates
  - Set API provider preferences
  - Check API status
  - Generate missing summaries

## Troubleshooting

If you encounter API issues:

1. Check your API keys are correctly set in the `.env` file
2. Use the API Diagnostics tool in the Admin Panel to test connections
3. Look at the `app.log` and `summarizer.log` files for detailed error messages

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE) 