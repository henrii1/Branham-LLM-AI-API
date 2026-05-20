
python scripts/deploy_cloudrun.py

curl -N -X POST https://api.branhamsermons.ai/api/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer b6766b2e-9a26-4342-9bef-5da4ad67e51c" \
  -d '{"conversation_id": "test-001", "query": "Who is William Branham?"}'


KMP_DUPLICATE_LIB_OK=TRUE uv run uvicorn branham_model_api.api.main:app --reload --port 8000


KEY="$(uv run python - <<'PY'
from branham_model_api.api.routes import chat
print(chat._get_expected_chat_bearer_key())
PY
)"

curl -N \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"conversation_id":"term-stream-1","query":"What is balm in Gilead?","history_window":[],"conversation_summary":""}' \
  http://127.0.0.1:8000/api/chat