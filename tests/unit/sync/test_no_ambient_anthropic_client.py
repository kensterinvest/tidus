import pathlib
import re

SYNC_DIR = pathlib.Path(__file__).resolve().parents[3] / "tidus" / "sync"
# zero-arg AsyncAnthropic()/Anthropic() with no api_key= inside the parens
_BARE = re.compile(r"\b(?:Async)?Anthropic\(\s*\)")
_NO_KEY = re.compile(r"\b(?:Async)?Anthropic\((?![^)]*api_key)[^)]*\)")


def test_no_zero_arg_or_keyless_anthropic_in_sync():
    offenders = []
    for py in SYNC_DIR.rglob("*.py"):
        src = py.read_text(encoding="utf-8")
        for m in list(_BARE.finditer(src)) + list(_NO_KEY.finditer(src)):
            offenders.append(f"{py.relative_to(SYNC_DIR)}: {m.group(0)}")
    assert not offenders, (
        "tidus/sync must construct Anthropic only with an explicit api_key= "
        "(via build_sync_anthropic_client). Offenders:\n" + "\n".join(offenders)
    )
