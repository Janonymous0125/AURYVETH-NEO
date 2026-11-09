import json
import tempfile
import unittest
from pathlib import Path

from neural_tokenizer import LayeredTokenizer, RegistryError, ContractMismatch


class TokenizerTests(unittest.TestCase):
    def setUp(self):
        self.t = LayeredTokenizer()

    def test_default_byte_ids_identical_to_legacy(self):
        s = 'Hello! 你好 👩\u200d👩\u200d👧\u200d👧 \U0001f600\n'
        self.assertEqual(self.t.encode(s), list(s.encode('utf-8')))
        self.assertEqual(self.t.encode('A', add_bos=True, add_eos=True), [256, 65, 257])
        self.assertEqual(self.t.decode(self.t.encode(s)), s)
        self.assertEqual(self.t.vocab_size, 258)

    def test_roundtrip_with_bpe_and_literals(self):
        corpus = ['banana banana banana banana', 'bandana banana', '中文中文中文', '你好世界你好世界']
        n = self.t.train_bpe(corpus, max_merges=16, min_frequency=2)
        self.assertGreater(n, 0)
        self.t.register_literal_token('🙃')
        for s in corpus + ['中文 ↔ English 🌱', '👩\u200d👩\u200d👧\u200d👧', '', 'a\u0308']:
            ids = self.t.encode(s)
            self.assertEqual(self.t.decode(ids), s)
            self.assertTrue(all(i < self.t.vocab_size for i in ids))

    def test_literal_registers_new_model_token_only_explicitly(self):
        self.t.observe('Mochi plays with 猫 😋', source='owner:1')
        self.assertEqual(self.t.vocab_size, 258)
        new_id = self.t.register_literal_token('Mochi')
        self.assertEqual(new_id, 258)
        self.assertEqual(self.t.encode('Mochi'), [258])
        self.assertEqual(self.t.register_literal_token('Mochi'), new_id)
        self.assertEqual(self.t.decode([258]), 'Mochi')

    def test_word_ids_stable_across_observation_reload_and_no_vocab_mutation(self):
        self.t.observe('Mochi likes 中文 and 😋', source='example:A')
        a = {x['surface']: x['id'] for x in self.t.word_entries()}
        self.assertIn('Mochi', a)
        self.assertIn('中文', a)
        self.assertIn('😋', a)
        self.t.observe('Mochi 中文', source='example:B')
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'registry.json'
            self.t.save(p)
            other = LayeredTokenizer.load(p)
            b = {x['surface']: x['id'] for x in other.word_entries()}
            self.assertEqual(a, b)
            self.assertEqual(other.vocab_size, 258)

    def test_analysis_alignment_lossless_with_complex_unicode(self):
        s = 'Hello, 中文!👩\u200d👩\u200d👧\u200d👧\nOther sentence.'
        report = self.t.analyze(s)
        data = s.encode('utf-8')
        self.assertEqual(b''.join(data[x['start_byte']:x['end_byte']] for x in report['tokens']), data)
        self.assertEqual(''.join(x['text'] for x in report['words']), s)
        self.assertEqual(''.join(x['text'] for x in report['sentences']), s)
        self.assertEqual(report['tokens'][-1]['end_byte'], len(data))

    def test_sentence_units_separate_from_lm_vocabulary(self):
        self.t.observe('First sentence. Second sentence.', source='example:A', remember_sentences=True)
        self.assertEqual(self.t.vocab_size, 258)
        self.assertGreaterEqual(len(self.t.sentence_entries()), 2)
        self.assertTrue(all(x['id'] >= 0 for x in self.t.sentence_entries()))

    def test_concept_candidates_require_distinct_attribution_and_approval(self):
        self.t.observe('Mochi is a cat.', source='source:A')
        c = self.t.concept_candidates()[0]
        with self.assertRaises(ValueError):
            self.t.approve_candidate(c['key'])
        self.t.observe('Mochi is a cat.', source='source:A')
        with self.assertRaises(ValueError):
            self.t.approve_candidate(c['key'])
        self.t.observe('Mochi is a cat.', source='source:B')
        self.assertEqual(len(self.t.relation_entries()), 0)
        self.t.approve_candidate(c['key'], approver='researcher')
        self.assertEqual(len(self.t.relation_entries()), 1)
        self.assertGreaterEqual(len(self.t.concept_entries()), 2)
        self.assertEqual(len(self.t.relation_entries()), 1)

    def test_chinese_candidate_and_rejection_retained(self):
        self.t.observe('猫是一种动物。', source='A')
        c = self.t.concept_candidates()[0]
        self.assertEqual((c['subject'], c['relation'], c['object']), ('猫', 'is_a', '动物'))
        self.t.reject_candidate(c['key'], reviewer='reviewer', reason='insufficient trust')
        self.t.observe('猫是一种动物。', source='B')
        with self.assertRaises(ValueError):
            self.t.approve_candidate(c['key'], approver='researcher')

    def test_candidate_survives_reload(self):
        self.t.observe('Mochi is a cat.', source='A')
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'registry.json'
            self.t.save(p)
            u = LayeredTokenizer.load(p)
            self.assertEqual(self.t.concept_candidates(), u.concept_candidates())
            u.observe('Mochi is a cat.', source='B')
            c = u.concept_candidates()[0]
            u.approve_candidate(c['key'], approver='reviewer')
            u.save(p)
            v = LayeredTokenizer.load(p)
            self.assertEqual(v.relation_entries(), u.relation_entries())

    def test_contract_changes_only_when_lm_vocabulary_changes(self):
        c = self.t.model_contract()
        self.t.observe('A new memory, 中文!', source='A')
        self.t.assert_model_contract(c)
        self.t.register_literal_token('Mochi')
        with self.assertRaises(ContractMismatch):
            self.t.assert_model_contract(c)

    def test_corrupt_registry_rejected_no_unsafe_code_execution(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'registry.json'
            self.t.save(p)
            raw = json.loads(p.read_text(encoding='utf8'))
            raw['data']['next_token_id'] = 999999
            p.write_text(json.dumps(raw), encoding='utf8')
            with self.assertRaises(RegistryError):
                LayeredTokenizer.load(p)

    def test_invalid_token_ids_rejected(self):
        with self.assertRaises(ValueError):
            self.t.decode([9999])
        with self.assertRaises(TypeError):
            self.t.encode(b'not str')
        with self.assertRaises(UnicodeError):
            self.t.encode('\ud800')

    def test_model_vocab_preserves_ids_across_save(self):
        self.t.train_bpe(['ababababab', 'ababababab'], max_merges=5)
        lid = self.t.register_literal_token('新词')
        before = self.t.encode('abab新词')
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'t.json'
            self.t.save(p)
            loaded = LayeredTokenizer.load(p)
            self.assertEqual(loaded.encode('abab新词'), before)
            self.assertEqual(loaded.register_literal_token('新词'), lid)
            self.assertEqual(loaded.model_contract(), self.t.model_contract())

    def test_no_automatic_registration_on_readonly_analysis(self):
        contract = self.t.model_contract()
        self.t.analyze('Mochi is a cat.')
        self.assertEqual(self.t.word_entries(), [])
        self.assertEqual(self.t.concept_candidates(), [])
        self.assertEqual(self.t.model_contract(), contract)


if __name__ == '__main__':
    unittest.main()

class RobustnessTests(unittest.TestCase):
    def test_unicode_corpus_and_alignment_with_learned_merges(self):
        import random
        t = LayeredTokenizer()
        t.train_bpe(['a b a b c', 'a b a b c', '猫猫猫猫🌏', 'Hello Hello Hello'], max_merges=20)
        t.register_literal_token('🌏')
        r = random.Random(713)
        symbols = ['a', ' ', '\n', '猫', '字', '🌏', '👨\u200d👩\u200d👧', 'a\u0301', 'مرحبا', '⚡']
        for _ in range(70):
            s = ''.join(r.choices(symbols, k=r.randint(0, 35)))
            a = t.analyze(s)
            self.assertEqual(t.decode(a['token_ids']), s)
            self.assertEqual(''.join(x['text'] for x in a['words']), s)
            self.assertEqual(''.join(x['text'] for x in a['sentences']), s)
            for kind in ('words', 'sentences'):
                for x in a[kind]:
                    self.assertEqual(s[x['start_char']:x['end_char']], x['text'])
                    for i in x['token_indices']:
                        tk = a['tokens'][i]
                        self.assertLess(tk['start_byte'], x['end_byte'])
                        self.assertGreater(tk['end_byte'], x['start_byte'])

    def test_previous_approval_does_not_change_neural_tokens(self):
        t = LayeredTokenizer()
        original = t.model_contract()
        t.observe('Mochi is a cat.', source='owner:A')
        t.observe('Mochi is a cat.', source='owner:B')
        t.approve_candidate(t.concept_candidates()[0]['key'], approver='reviewer')
        self.assertEqual(t.model_contract(), original)
        a = t.analyze('Mochi is a cat.')['concept_hypotheses'][0]
        self.assertEqual(a['status'], 'approved')
        self.assertEqual(a['relation_id'], 0)

    def test_two_sources_are_not_asserted_to_be_truth(self):
        t = LayeredTokenizer()
        t.observe('Mars is a jellyfish.', source='fiction:A')
        t.observe('Mars is a jellyfish.', source='fiction:B')
        self.assertEqual(t.concept_entries(), [])
        self.assertEqual(t.relation_entries(), [])
        self.assertEqual(t.concept_candidates()[0]['status'], 'pending')

    def test_literal_vocab_growth_requires_explicit_migration(self):
        t = LayeredTokenizer()
        bpe_count = t.train_bpe(['abcabcabcabc'], max_merges=3)
        self.assertGreaterEqual(bpe_count, 1)
        contract = t.model_contract()
        t.observe('neural memories', source='alpha')
        t.assert_model_contract(contract)
        t.train_bpe(['newnewnewnew'], max_merges=2)
        with self.assertRaises(ContractMismatch):
            t.assert_model_contract(contract)
