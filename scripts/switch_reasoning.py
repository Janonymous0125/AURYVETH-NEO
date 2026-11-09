"""Use a planner checkpoint to predict one switch transition at a time."""
import torch

from scripts.build_conversation_data import format_prompt


STATES = {'en': ('off', 'on'), 'zh': ('关', '开')}
COMMANDS = ('keep', 'toggle', 'set-off', 'set-on')
WORDS = {
    'en': ('keep it unchanged', 'toggle it', 'turn it off', 'turn it on'),
    'zh': ('保持状态不变', '切换状态', '将开关关闭', '将开关打开'),
}
TEMPLATES = {
    'en': (
        'The switch is {state}. Command: {action}. Give the resulting state, off or on.',
        'Current state: {state}. Apply this instruction: {action}. What is the new switch state?',
        'A light starts {state}. Next, {action}. Answer off or on.',
        'Initially the light is {state}. After you {action}, is it on or off?',
        'Begin with the switch {state}. Follow one command: {action}. Report its final state.',
        'Switch {state}. {action}. Result?',
        'Light {state}. Now {action}. Next state?',
        '{state} → {action}. Final state?',
    ),
    'zh': (
        '开关目前是{state}。指令：{action}。请回答结果是关还是开。',
        '当前状态：{state}。执行指令：{action}。开关的新状态是什么？',
        '灯一开始是{state}。接下来，{action}。只回答开或关。',
        '灯的初始状态是{state}。执行“{action}”后，它是开还是关？',
        '开关起初处于{state}状态。请{action}，然后报告它的最终状态。',
        '开关{state}。{action}。结果？',
        '灯是{state}。现在{action}。新状态？',
        '{state} → {action}。最终状态？',
    ),
}
MAX_COMMANDS = 16


def transition_prompt(state, command, language, variant):
    action = WORDS[language][COMMANDS.index(command)]
    content = TEMPLATES[language][variant].format(state=state, action=action)
    return format_prompt([{'role': 'user', 'content': content}])


@torch.inference_mode()
def deliberate_switch(model, tokenizer, state, commands, *, language='en', variant=0):
    """Feed each predicted state into the next model call and return the final label."""
    states = STATES[language]
    if state not in states or not 1 <= len(commands) <= MAX_COMMANDS or any(c not in COMMANDS for c in commands):
        raise ValueError('Use on, off, 开, or 关, followed by 1–16 commands: keep, toggle, set-off, set-on.')
    if model.training or model.config.backbone != 'recurrent' or not model.config.planner_steps:
        raise ValueError('Switch reasoning requires an evaluated recurrent planner checkpoint.')
    state_tokens = [tokenizer.encode(label) for label in states]
    if any(len(ids) != 1 for ids in state_tokens) or state_tokens[0] == state_tokens[1]:
        raise ValueError('This tokenizer cannot represent both switch labels as distinct single tokens.')
    labels = [ids[0] for ids in state_tokens]
    trace = []
    for command in commands:
        prompt = transition_prompt(state, command, language, variant)
        ids = torch.tensor([tokenizer.encode(prompt)], device=model.token_embedding.weight.device)
        logits, _ = model._recurrent_prefill(ids)
        scores = logits[0, labels]
        if not bool(torch.isfinite(scores).all()):
            raise ValueError('The planner returned invalid switch scores.')
        state = states[int(scores.argmax())]
        trace.append(state)
    return state, trace
