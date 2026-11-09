import sys
from types import SimpleNamespace

import torch

from scripts import chat
from scripts.switch_reasoning import deliberate_switch


def test_deliberation_feeds_each_model_prediction_into_the_next_prompt():
    prompts = []

    class Tokenizer:
        def encode(self, text):
            if text in ('off', 'on'):
                return [int(text == 'on')]
            prompts.append(text)
            return [2]

    scores = iter((torch.tensor([[0., 1.]]), torch.tensor([[1., 0.]])))
    model = SimpleNamespace(training=False, config=SimpleNamespace(backbone='recurrent', planner_steps=3),
                            token_embedding=SimpleNamespace(weight=torch.zeros(1)),
                            _recurrent_prefill=lambda _: (next(scores), None))
    # The fixture intentionally predicts a state change after "keep".
    # This catches a hidden rule or code path that ignores the model prediction.
    assert deliberate_switch(model, Tokenizer(), 'off', ['keep', 'keep']) == ('off', ['on', 'off'])
    assert 'switch is off' in prompts[0] and 'switch is on' in prompts[1]


def test_opt_in_think_turn_does_not_feed_switch_labels_to_chat_history(monkeypatch, capsys):
    baseline, planner = object(), object()
    monkeypatch.setattr(sys, 'argv', ['chat', '--checkpoint', 'baseline.pt', '--think-manifest', 'think.json'])
    monkeypatch.setattr(chat, 'SentencePieceTokenizer', lambda _: object())
    monkeypatch.setattr(chat, 'verify_manifest', lambda _: (
        {'model': 'planner.pt'}, {'task': 'switch-transitions-v1', 'cli_qualified': True, 'prompt_variant': 5}))
    monkeypatch.setattr(chat.IntegratedLM, 'load', lambda name, _: SimpleNamespace(
        eval=lambda: baseline if name == 'baseline.pt' else planner))
    inputs = iter(('Hello', '/think off toggle', '/think off', 'Follow-up', '/quit'))
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))

    def deliberate(model, tokenizer, initial, commands, *, language, variant):
        assert model is planner and initial == 'off' and commands == ['toggle']
        assert language == 'en' and variant == 5
        return 'on', ['on']

    def reply(model, tokenizer, messages, **options):
        assert model is baseline
        if messages[-1]['content'] == 'Follow-up':
            assert [m['content'] for m in messages] == ['Hello', 'chat response', 'Follow-up']
        return 'chat response'

    monkeypatch.setattr(chat, 'deliberate_switch', deliberate)
    monkeypatch.setattr(chat, 'reply', reply)
    chat.main()
    assert 'Assistant: on' in capsys.readouterr().out
