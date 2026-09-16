from llmcomm.pipeline.sentence_splitter import StreamSentenceSplitter, clean_for_tts


def test_stream_split_korean():
    s = StreamSentenceSplitter(min_chars=4)
    out = []
    for d in ["안녕하", "세요. 오늘 날", "씨가 좋네요! 산책", " 갈까요?", " 네"]:
        out += s.feed(d)
    out += s.flush()
    assert out == ["안녕하세요.", "오늘 날씨가 좋네요!", "산책 갈까요?", "네"]


def test_short_sentence_merges_forward():
    s = StreamSentenceSplitter(min_chars=6)
    out = s.feed("네. 알겠습니다. ")
    assert out == ["네. 알겠습니다."]


def test_clean_for_tts_strips_emoji_and_markdown():
    assert clean_for_tts("안녕! 오늘 기분 좋아 😊") == "안녕! 오늘 기분 좋아"
    assert clean_for_tts("**중요**: `코드` 확인") == "중요: 코드 확인"
    assert clean_for_tts("- 첫째\n- 둘째") == "첫째\n둘째"


def test_splitter_drops_emoji_only_fragment():
    s = StreamSentenceSplitter(min_chars=4)
    out = s.feed("좋은 아침이야. 😊😊. 오늘 뭐 해?") + s.flush()
    assert out == ["좋은 아침이야.", "오늘 뭐 해?"]
