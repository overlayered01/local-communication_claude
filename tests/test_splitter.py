from llmcomm.pipeline.sentence_splitter import StreamSentenceSplitter


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
