"""LLM transport of the search module (T2-41 S3): settings (config: LLMConfig, LLMKeyMissing), the Phase 1 client
copies (base: LLMClient, LLMResult, STATUS_*, MockClient; hcx: HCXClient), the other clients (openai: OpenAIClient,
clova_rerank: ClovaReranker) and the provider table (providers: PROVIDERS, build_client, clova_chat).

Keys come in as arguments (LLMConfig.api_key); nothing here reads .env or the environment. urllib.request, openai and
the CLOVA chat SDK are imported only in this folder (providers.py; tests/test_layout.py). llm imports no other part of kosis_rag.
"""
