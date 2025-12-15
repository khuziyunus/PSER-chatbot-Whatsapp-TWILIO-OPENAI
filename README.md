# Twilio-OpenAI-WhatsApp-Bot

This repo contains code to build a chatbot in Python that runs in WhatsApp and answers user messages using OpenAI API. Developed with FastAPI and we use Twilio for Business WhatsApp integration. The source code is available on GitHub under an open-source license.

## Tech stack:
- Python
- Docker
- FastAPI
- Twilio
- OpenAI API
- Redis

## Getting Started

To get started, follow these steps:

1. Clone the repository to your local machine:
   ```bash
   git clone git@github.com:Shakurova/Twilio-OpenAI-WhatsApp-Bot.git
   cd Twilio-OpenAI-WhatsApp-Bot/
   ```

2. Setting up Twilio

   - **Twilio Sandbox for WhatsApp**: Start by setting up the Twilio Sandbox for WhatsApp to test your application. This allows you to send and receive messages using a temporary WhatsApp number provided by Twilio. Follow the steps in the Twilio Console under the "Messaging" section to configure the sandbox. You can find detailed instructions in the [Twilio Blog Guide](https://www.twilio.com/en-us/blog/ai-chatbot-whatsapp-python-twilio-openai).

   - **Moving to Production**: Once you have tested your application in the sandbox environment and are ready to go live, you can set up a Twilio phone number for production use. This involves purchasing a Twilio number and configuring it to handle WhatsApp messages. Refer to [Twilio Guide](https://www.twilio.com/docs/whatsapp) for more information on transitioning to a production environment.

3. Make sure you have docker and redis installed on your machine.

   For macOS:
   Install Redis using Homebrew:
   ```bash
   brew install redis
   ```
   Start Redis:
   ```bash
   brew services start redis
   ```

   Install Docker via Homebrew:
   ```bash
   brew install --cask docker
   Open Docker Desktop and make sure it’s running.
   ```
   Verify installation:
   ```bash
   docker --version
   ```

4. Create a `.env` file in the project directory and set your OpenAI API key, Twilio account details, and (optionally) the knowledge-base file path as environment variables. By default, the app uses `app/data/PSER_info.txt`.
   ```plaintext
    TWILIO_WHATSAPP_NUMBER=<your Twilio phone number>
    TWILIO_ACCOUNT_SID=<your Twilio account SID>
    TWILIO_AUTH_TOKEN=<your Twilio auth token>
    OPENAI_API_KEY=<your OpenAI API key>
    REDIS_HOST=<your redis host>
    REDIS_PORT=<your redis port>
    REDIS_PASSWORD=<your redis password>
    DATA_RAG=<optional absolute or relative path to a .txt knowledge base file>
   ```
   Optional tuning flags (set only if needed):
   - `ENABLE_CHAT_HISTORY=true|false` — keep Redis conversation memory.
   - `ENABLE_CONTEXTUALIZER=true|false` — rewrite follow-up questions.
   - `DEBUG_RAG_CHUNKS=true|false` — log retrieved context chunks.
   - `RAG_PROVIDER=openai|groq`, `RAG_OPENAI_MODEL`, `RAG_GROQ_MODEL`, `RAG_TEMPERATURE`, `RAG_MAX_TOKENS`, `GROQ_API_KEY`.

5. Build and start the chatbot containers by running:
   ```bash
   docker-compose up --build -d
   ```

## Requirements

- Python 3.10+
- Docker and Docker Compose
- Redis instance (local or remote)
- Twilio WhatsApp Sandbox or a WhatsApp-enabled Twilio number
- OpenAI API key (`OPENAI_API_KEY`)
- Optional: Google Cloud Translate v3 (`GOOGLE_PROJECT_ID`, `GOOGLE_APPLICATION_CREDENTIALS`) for language detection/translation

## How to Run

### Docker Compose (recommended)

1. Ensure `.env` contains all required variables.
2. Run:
   ```bash
   docker-compose up --build -d
   ```
3. Open API docs:
   ```
   http://localhost:3002/docs
   ```
4. Configure Twilio webhook to:
   ```
   http://<your-host>:3002/whatsapp-endpoint
   ```

### Local (no Docker)

1. Create and activate a virtual environment.
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```
2. Install dependencies.
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Ensure Redis is running and `.env` is set.
4. Start the API server.
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 3002 --reload
   ```
   Or with Gunicorn:
   ```bash
   gunicorn app.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:3002
   ```
5. Verify:
   ```
   http://localhost:3002/docs
   ```


## Language Detection & Translation

- Incoming WhatsApp messages are automatically processed with language detection.
- The message is translated to English and passed into the RAG pipeline for answering.
- The final answer is translated back to the original detected language and returned to the user on WhatsApp.

### Providers

- Primary: Google Cloud Translate v3 via `TranslationServiceClient` when `GOOGLE_PROJECT_ID` and `GOOGLE_APPLICATION_CREDENTIALS` are set.
- Fallback: OpenAI (`gpt-3.5-turbo-0125`) used for detection/translation when Google Translate is not configured.

### Quick Verification

- After `docker-compose up`, open `http://localhost:3002/docs` to verify the FastAPI server.
- Point Twilio webhook to `http://<host>:3002/whatsapp-endpoint`.

### Notes

- The FAISS index is built in-memory from `DATA_RAG` (default `app/data/PSER_info.txt`). Use `DATA_RAG` to override with your own `.txt` corpus.
- Ensure your Google service account has the Cloud Translation API enabled in the target project.
