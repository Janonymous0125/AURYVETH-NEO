from scripts import neo_ollama


def user(text):
    return [{'role': 'user', 'content': text}]


def test_all_chat_turns_return_model_output_without_replacements(monkeypatch):
    calls = []
    outputs = {'hi': '', '你好呀': '我是' + '一个' * 25, '为什么？': '这是模型生成的回复。'}

    def reply(_model, _tokenizer, history, **_options):
        calls.append(history)
        return outputs[history[-1]['content']]

    monkeypatch.setattr(neo_ollama, 'reply', reply)
    models = (None, None, None, 5)
    for prompt in ('hi', '你好呀'):
        assert neo_ollama.answer(models, user(prompt), {}) == (outputs[prompt], '')
    history = [*user('hi'), {'role': 'assistant', 'content': ''}, *user('为什么？')]
    assert neo_ollama.answer(models, history, {}) == (outputs['为什么？'], '')
    assert calls == [user('hi'), user('你好呀'), history]


def test_trace_displays_model_predictions(monkeypatch):
    # The fixture predicts an unexpected state after keep so the test detects
    # a trace assembled from hardcoded switch rules.
    monkeypatch.setattr(neo_ollama, 'deliberate_switch', lambda *args, **options: ('on', ['on', 'on']))
    text, trace = neo_ollama.answer((None, object(), None, 5), user('think: off keep keep'), {})
    assert text == 'on'
    assert trace.splitlines() == ['Start: off', '1. keep → on', '2. keep → on']
