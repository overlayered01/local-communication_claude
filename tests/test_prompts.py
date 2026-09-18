from llmcomm.core.prompts import Style, build_messages, is_polite, list_prompts, load_prompt, split_sentences, style_metrics
from llmcomm.core.types import Message


def test_all_prompt_presets_load_and_compose():
    names = list_prompts()
    assert "default" in names and "voice_banmal" in names
    for n in names:
        p = load_prompt(n)
        assert p.persona.strip()
        sysm = p.system()
        assert p.persona.strip().splitlines()[0] in sysm
    # rules are appended to the system text
    assert "반말" in load_prompt("voice_banmal").system()
    assert "존댓말" in load_prompt("voice_jondae").system()
    assert "이모지" in load_prompt("voice_banmal").system()


def test_default_prompt_matches_round1_text():
    assert load_prompt("default").system() == "당신은 친근한 한국어 대화 상대입니다. 짧고 자연스럽게 답하세요."


def test_build_messages_prepends_system_and_few_shot_and_drops_client_system():
    hist = [Message("system", "ignored"), Message("user", "안녕")]
    msgs = build_messages("persona_friend", hist)
    assert msgs[0].role == "system" and "모모" in msgs[0].content
    assert [m.role for m in msgs[1:]] == ["user", "assistant", "user", "assistant", "user"]
    assert msgs[-1].content == "안녕"
    assert build_messages("default", hist, system_override="X")[0].content == "X"


def test_register_detection():
    assert is_polite("오늘 날씨 좋네요.") is True
    assert is_polite("내일 만나요!") is True
    assert is_polite("점심 뭐 먹을까?") is False
    assert is_polite("그래, 알겠어.") is False
    assert is_polite("🙂") is None


def test_style_metrics_counts_violations():
    st = Style(register="banmal", max_sentences=2, no_emoji=True, no_markdown=True)
    good = style_metrics("야 그거 좋다. 나도 갈래.", st)
    assert good["n_sentences"] == 2 and good["style_violations"] == 0
    bad = style_metrics("좋은 생각이에요. **정말** 그래요 😊. 같이 가요.", st)
    # 3 polite sentences + over max + emoji + markdown
    assert bad["polite_sentences"] == 3 and bad["emoji_count"] == 1 and bad["markdown_count"] >= 1
    assert bad["style_violations"] == 3 + 1 + 1 + 1
    assert style_metrics("怎么 오늘은", Style())["hanja_chars"] == 2


def test_split_sentences():
    assert split_sentences("안녕! 잘 지냈어? 응.") == ["안녕!", "잘 지냈어?", "응."]
