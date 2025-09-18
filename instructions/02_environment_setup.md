# Environment Setup Instructions

## Virtual Environment

### Activate Environment

**Important:** Always activate the virtual environment before running any scripts.

```powershell
.\venv\Scripts\activate
```

**Location:** The virtual environment is located at `.\venv\` in the root directory.

## Environment Variables

Create a `.env` file in the `.\main\` directory (formerly `.\new_scripts\`) with the following variables:

```env
LM_STUDIO_HOST=http://127.0.0.1
LM_STUDIO_PORT=1234
LM_STUDIO_MODEL=<whatever_model_you_want_from_lmstudio>
```

### Variable Descriptions

- **LM_STUDIO_HOST:** The host address for LM Studio API endpoint
- **LM_STUDIO_PORT:** The port number for LM Studio API
- **LM_STUDIO_MODEL:** The specific model name you want to use from LM Studio

### Example Configuration

```env
LM_STUDIO_HOST=http://127.0.0.1
LM_STUDIO_PORT=1234
LM_STUDIO_MODEL=llama-3.2-8b-instruct
```

## LM Studio Setup

1. **Install LM Studio** - Download and install LM Studio locally
2. **Load Model** - Load your preferred language model in LM Studio
3. **Start Server** - Enable the local API server in LM Studio
4. **Configure Port** - Ensure LM Studio is running on port 1234 (or update .env accordingly)
5. **Test Connection** - The scripts will automatically test the LM Studio connection

## Dependencies

Ensure all required packages are installed:

```powershell
# After activating venv
pip install -r requirements.txt
```
