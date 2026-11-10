import sqlite3
import tempfile
import unittest
from pathlib import Path
from neo_chat_experimental import connect,remember,list_memories,log_turn,construct_messages,recent_turns

class MemoryTests(unittest.TestCase):
    def test_restart_persistence(self):
        with tempfile.TemporaryDirectory() as d:
            db=str(Path(d)/'memory.sqlite3')
            one=connect(db)
            remember(one,'My favorite color is teal')
            log_turn(one,'user','hello')
            one.close()
            two=connect(db)
            self.assertEqual(list_memories(two),['My favorite color is teal'])
            self.assertEqual(recent_turns(two),[{'role':'user','content':'hello'}])
            self.assertIn('My favorite color is teal',construct_messages(two,'remember?')[0]['content'])
            two.close()
    def test_duplicates_and_invalid_memory(self):
        with tempfile.TemporaryDirectory() as d:
            c=connect(str(Path(d)/'data.db'))
            self.assertEqual(remember(c,'a'),remember(c,'a'))
            self.assertEqual(len(list_memories(c)),1)
            for val in ('',' '*4,'x'*4001):
                with self.assertRaises(ValueError):remember(c,val)
            c.close()
    def test_sql_injection_data_only(self):
        with tempfile.TemporaryDirectory() as d:
            c=connect(str(Path(d)/'data.db'))
            remember(c,"'); DROP TABLE memories; --")
            self.assertEqual(len(list_memories(c)),1)
            c.close()
    def test_conversation_control(self):
        with tempfile.TemporaryDirectory() as d:
            c=connect(str(Path(d)/'data.db'))
            remember(c,'fact')
            log_turn(c,'user','private')
            c.execute('DELETE FROM turns');c.commit()
            self.assertEqual(recent_turns(c),[])
            self.assertEqual(list_memories(c),['fact'])
            c.close()

if __name__=='__main__': unittest.main()