"""Regression for Unicode text separators and accurate corruption messages."""
import json
import pytest
from model import IntegratedConfig
from neural_tokenizer import LayeredTokenizer
from scripts.check_jsonl import check
from scripts.train import load_rows


def config():
    return IntegratedConfig(vocab_size=258,block_size=192,d_model=32,n_layers=1,n_heads=4,
                             n_kv_heads=2,ffn_hidden=64,memory_tokens=64,active_memories=2)


@pytest.mark.parametrize('separator',['\u2028','\u2029','\u0085'])
def test_unicode_text_separator_does_not_split_json_record(tmp_path,separator):
    tok=LayeredTokenizer()
    original={'context':['A true memory'+separator+'continues here.'],
              'prompt':'What happened? '+separator+'Explain.',
              'answer':'The event'+separator+'continued.'}
    file=tmp_path/'data.jsonl'
    file.write_text(json.dumps(original,ensure_ascii=False)+'\n',encoding='utf-8')
    assert len(file.read_text(encoding='utf-8').splitlines())>1  # old bug reproduced
    assert check(file)['examples']==1
    rows=load_rows(file,tok,config())
    assert len(rows)==1
    assert rows[0][2][0] == tok.encode(original['context'][0])


def test_invalid_jsonl_reports_actual_line_without_modifying(tmp_path):
    file=tmp_path/'damaged.jsonl'
    valid=json.dumps({'context':[],'prompt':'P','answer':'A'})
    file.write_text(valid+'\n'+ '{"context": ["broken\n',encoding='utf-8')
    before=file.read_bytes()
    with pytest.raises(ValueError,match=r'physical line 2, column') as error:
        load_rows(file,LayeredTokenizer(),config())
    assert 'Invalid control character' in str(error.value)
    with pytest.raises(ValueError,match=r'physical line 2, column'):
        check(file)
    assert before==file.read_bytes()


def test_two_records_have_expected_physical_lines(tmp_path):
    file=tmp_path/'multi.jsonl'
    lines=[{'context':[],'prompt':'First \u2028 question','answer':'yes'},
           {'context':[],'prompt':'Second','answer':'no'}]
    file.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in lines),encoding='utf-8')
    assert check(file)['examples']==2
    assert len(load_rows(file,LayeredTokenizer(),config()))==2
