MODEL_TYPES = {
    "google_genai": {
        "chat_models": [
            "gemini-1.5-pro",
            "gemini-2.0-flash-lite",
            "gemini-2.0-flash-full",
            "gemini-2.0-flash-large",
            "gemini-2.0-flash-large-v2",
            "gemini-2.0-flash-large-v2-1",
            "gemini-2.0-flash-large-v2-2",
            "gemini-2.0-flash-large-v2-3",
            "gemini-2.0-flash-large-v2-4",
            "gemini-2.0-flash-large-v2-5",
            "gemini-2.0-flash-large-v2-6",
            "gemini-2.0-flash-large-v2-7",
            "gemini-2.0-flash-large-v2-8",
            "gemini-2.0-flash-large-v2-9",
            "gemini-2.0-flash-large-v2"
        ],
        "default": "gemini-2.0-flash-lite",
        "langchain_module": "langchain_google_genai"
    },
    "openai": {
        "chat_models":
            [
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-16k",
                "gpt-4",
                "gpt-4-32k",
                "gpt-4-1106-preview",
                "gpt-4-vision-preview",
                "gpt-4-1106-preview",
                "gpt-4-vision-preview",
                "gpt-4-1106-preview",
                "gpt-4-vision-preview",
                "gpt-4-1106-preview",
                "gpt-4-vision-preview",
                "gpt-4-1106-preview",
                "gpt-4-vision-preview",
                "gpt-4-1106-preview",
                "gpt-4-vision-preview",
                "gpt-4-1106-preview",
                "gpt-4-vision-preview",
                "gpt-4-1106-preview",
                "gpt-4-vision-preview"
            ],
        "default": "gpt-3.5-turbo",
        "langchain_module": "langchain_openai"
    }
}
