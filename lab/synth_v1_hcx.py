"""HCX share of the v1 training questions (plan 47): one question per slot card, written by HCX.

Reads data/lab/synth/v1/cards_hcx.jsonl (from `python lab/synth_v1.py plan`) and appends to v1/hcx.jsonl, so a stopped
run resumes where it left off. Several cards go in one call (--batch, default 10). The model sees only the slot card
(never a table name or a golden question). Key: HCX_API_KEY from .env through the Phase 1 HCXClient; never printed.

    python lab/synth_v1_hcx.py --limit 20      first look (2 calls)
    python lab/synth_v1_hcx.py                 the rest
    python lab/synth_v1_hcx.py --model HCX-007 --thinking none     another model (default: configs/model.yaml)
    python lab/synth_v1_hcx.py --cards data/lab/synth/v1/cards_claude.jsonl --out data/lab/synth/v1/hcx_extra.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "vendor" / "phase1" / "src", ROOT / "vendor" / "phase1"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

V1 = ROOT / "data" / "lab" / "synth" / "v1"

SYSTEM = """너는 한국 국가통계포털(KOSIS) 사용자가 검색창이나 챗봇에 입력할 법한 질문을 쓰는 사람이다.
각 카드는 질문 하나의 조건이다. 카드마다 한국어 질문을 정확히 한 문장 쓴다.
규칙:
- 카드의 의도, 세는 대상(또는 개념), 분류, 지역, 기간, 주기를 모두 질문에 담는다. 카드에 없는 조건(연도, 지역 등)은 넣지 않는다.
- '기간: 언급 없음'이면 연도 숫자를 쓰지 않는다. '최근'이면 '최근', '요즘' 같은 말로 쓴다.
- 연도는 숫자 4자리로 쓴다(예: 2015년). 지역은 줄임말(서울, 경기)도 좋다.
- '개념'이 있으면 지표 이름을 쓰지 말고 현상이나 궁금증으로 묻는다.
- 세는 대상이 어색한 통계 용어면 일상어로 바꿔도 되지만 뜻은 같아야 한다.
- 말투 지시를 따른다. 질문마다 표현을 다르게 하고, 60자를 넘지 않는다.
출력: 카드마다 한 줄, '번호<TAB>질문' 형식만. 설명은 쓰지 않는다."""


def chunks(xs: list, n: int):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def parse(text: str, ids: list[str]) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        m = re.match(r"^\s*(\d+)\s*[\t.:)\]]\s*(.+?)\s*$", line)
        if m and 1 <= int(m.group(1)) <= len(ids):
            q = m.group(2).strip().strip('"').strip()
            if q:
                out[ids[int(m.group(1)) - 1]] = q
    return out


def make_client(model: str | None, thinking: str, max_tokens: int):
    """Phase 1 HCXClient (urllib only, the client the PC already runs): CLOVA Studio v3 chat-completions with the model
    name, answer length and (HCX-007 only) reasoning effort overridden; it retries 429 / 5xx itself."""
    from kosis_chat.config import load_env, load_settings
    from kosis_chat.llm.model import HCXClient

    load_env(ROOT / ".env")
    st = load_settings(backend="hcx", model_name=model, thinking_level=thinking, max_tokens=max_tokens)
    return HCXClient(st)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="HCX share of synth_v1 (plan 47)")
    ap.add_argument("--cards", type=Path, default=V1 / "cards_hcx.jsonl")
    ap.add_argument("--out", type=Path, default=V1 / "hcx.jsonl")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="cards to do in this run (0 = all left)")
    ap.add_argument("--sleep", type=float, default=0.5, help="seconds between calls")
    ap.add_argument("--dry-run", action="store_true", help="print the first prompt, call nothing")
    ap.add_argument("--model", default=None, help="HCX model name, e.g. HCX-007 (default: configs/model.yaml)")
    ap.add_argument("--thinking", default="none", choices=("none", "low", "medium", "high"),
                    help="HCX-007 reasoning effort (sent only to HCX-007; its thinking tokens count toward max_tokens)")
    ap.add_argument("--max-tokens", type=int, default=2048, help="answer tokens per call (10 questions need ~600)")
    args = ap.parse_args(argv)

    cards = [json.loads(x) for x in args.cards.read_text(encoding="utf-8").splitlines() if x.strip()]
    done = set()
    if args.out.exists():
        done = {json.loads(x)["id"] for x in args.out.read_text(encoding="utf-8").splitlines() if x.strip()}
    todo = [c for c in cards if c["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"cards {len(cards)}, done {len(done)}, this run {len(todo)}")
    if not todo:
        return 0

    def prompt(batch):
        return "\n".join(f"{i}\t{c['card']}" for i, c in enumerate(batch, 1))

    if args.dry_run:
        print(SYSTEM + "\n---\n" + prompt(todo[: args.batch]))
        return 0

    client = make_client(args.model, args.thinking, args.max_tokens)
    model_name = client.settings.model_name
    print(f"model {model_name}" + (f", thinking {args.thinking}" if client.is_reasoning else ""))
    wrote = missing = failed = 0
    t0 = time.time()
    with args.out.open("a", encoding="utf-8") as f:
        for n, batch in enumerate(chunks(todo, args.batch), 1):
            ids = [c["id"] for c in batch]
            got = {}
            for attempt in range(3):
                res = client.generate(SYSTEM, prompt(batch))  # 429 / 5xx are retried inside
                if res.status == "ok" and res.text:
                    got = parse(res.text, ids)
                    break
                print(f"  call {n} attempt {attempt + 1}: {res.status} {res.error or ''}"[:200])
                time.sleep(5 * (attempt + 1))
            else:
                failed += 1
            for i in ids:
                if i in got:
                    f.write(json.dumps({"id": i, "question": got[i], "generator": "hcx", "model": model_name},
                                       ensure_ascii=False) + "\n")
                    wrote += 1
                else:
                    missing += 1
            f.flush()
            if n % 20 == 0:
                print(f"  {n} calls, {wrote} questions, {time.time() - t0:.0f}s")
            time.sleep(args.sleep)
    print(f"wrote {wrote}, missing {missing} (rerun to retry), failed calls {failed} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
