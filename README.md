# Multi-PDF Research Paper QA Assistant

This application allows you to upload multiple PDF research papers and ask questions about their content. The system uses the Landing AI API to extract text, tables, and figures from the PDFs, and leverages OpenAI's GPT-4o model to generate accurate answers with visual evidence.

![App Screenshot](https://via.placeholder.com/800x400?text=PDF+QA+Assistant)

## Features

- **Multiple PDF Support**: Upload several PDF files simultaneously
- **Document Analysis**: Extracts text, tables, and figures from PDFs using Landing AI API
- **Question Answering**: Ask questions about the content of your PDFs
- **Visual Evidence**: See exactly where in the document the answers come from with highlighted bounding boxes
- **Reasoning Transparency**: Includes detailed reasoning explaining how the answer was derived
- **Batch Processing**: Efficiently processes large PDFs in manageable chunks
- **Chat History**: Maintains conversation history for reference

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/multi-pdf-qa-assistant.git
cd multi-pdf-qa-assistant
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root directory with your API keys:
```
LANDING_API_KEY=your_landing_ai_api_key
OPENAI_API_KEY=your_openai_api_key
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app_osama_fix.py
```

2. Open the provided URL in your web browser (typically http://localhost:8501)

3. Upload one or more PDF research papers using the file uploader

4. Wait for the system to precompute evidence from the PDFs

5. Enter your question in the chat input box

6. View the answer, reasoning, and supporting evidence highlighted in the PDF pages

## Requirements

- Python 3.7+
- Landing AI API key (for document analysis)
- OpenAI API key (for question answering)

## Dependencies

- `streamlit`: Web application framework
- `openai`: OpenAI API client for GPT-4o
- `PyPDF2`: PDF manipulation
- `opencv-python`: Image processing
- `Pillow`: Image handling
- `requests`: HTTP requests
- `fpdf`: PDF generation
- `PyMuPDF`: PDF rendering
- `python-dotenv`: Environment variable management
- `numpy`: Numerical operations

## How It Works

1. **PDF Processing**: The application splits PDFs into chunks of 3 pages and processes them using the Landing AI API
2. **Evidence Extraction**: Text, tables, and figures are extracted with their positions (bounding boxes)
3. **Question Analysis**: When a user asks a question, the system searches through the precomputed evidence
4. **Answer Generation**: GPT-4o analyzes the relevant evidence and generates a comprehensive answer
5. **Visual Evidence**: The system highlights the exact locations in the PDF where the answer was found
6. **Result Presentation**: The answer, reasoning, and annotated PDF pages are displayed to the user

## Examples

Ask questions like:
- "What are the main findings of this research?"
- "What methodology was used in the study?"
- "Compare the results across these papers."
- "What are the limitations mentioned in the research?"

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- [Landing AI](https://landing.ai/) for their document analysis API
- [OpenAI](https://openai.com/) for their GPT-4o model
- [Streamlit](https://streamlit.io/) for the web application framework

## Contact

For questions or support, please open an issue in the GitHub repository.
