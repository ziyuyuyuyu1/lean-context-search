"""CPU-only CLI destination checks; no cache or inference may be reached."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lean_context_search import cli, encoder, qwen


class OutputPreflight(unittest.TestCase):
    def test_invalid_fresh_limit_fails_before_encoder_or_cache(self):
        for limit in ('0', '-1'):
            with self.subTest(limit=limit), \
                 patch.object(cli, 'ExactRetriever') as cache, \
                 patch.object(encoder, 'encode_local') as encode, \
                 contextlib.redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as caught:
                    cli.main(['search', '--documents', 'unused', '--query', 'fixture',
                              '--encoder', 'unused', '--encoder-contract', 'unused',
                              '--top-k', limit])
                self.assertEqual(caught.exception.code, 2)
                self.assertIn('top_k must be a positive integer', stderr.getvalue())
                cache.assert_not_called()
                encode.assert_not_called()

    def test_bad_destinations_fail_before_inputs_or_scoring(self):
        actions = {
            'retrieve': ['--documents', 'unused', '--queries', 'unused'],
            'reorder': ['--documents', 'unused', '--queries', 'unused', '--pools', 'unused'],
            'search': ['--documents', 'unused', '--query', 'fixture', '--encoder', 'unused',
                       '--encoder-contract', 'unused'],
            'rerank': ['--pools', 'unused', '--query-file', 'unused', '--model', 'unused',
                       '--passages', 'unused'],
            'rerank-cached': ['--pools', 'unused', '--features', 'unused', '--passages', 'unused'],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / 'existing.jsonl'
            existing.write_text('preserve this result\n')
            dangling = root / 'dangling.jsonl'
            dangling.symlink_to(root / 'absent-target')
            cases = [(existing, 'output already exists'),
                     (dangling, 'output already exists'),
                     (root / 'absent-directory' / 'output.jsonl', 'output directory'),
                     (existing / 'output.jsonl', 'output directory')]
            for action, arguments in actions.items():
                for destination, message in cases:
                    with self.subTest(action=action, destination=str(destination)), \
                         contextlib.ExitStack() as stack:
                        spies = [stack.enter_context(patch.object(module, name,
                                  side_effect=AssertionError(f'{name} was called before rejection')))
                                 for module, name in [(cli, 'ExactRetriever'), (cli, 'Cache'),
                                     (cli, 'read_jsonl'), (encoder, 'encode_local'),
                                     (qwen, 'score_live'), (qwen, 'score_saved_features'),
                                     (qwen, 'verify_passages')]]
                        stderr = stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                        with self.assertRaises(SystemExit) as caught:
                            cli.main([action, *arguments, '--output', str(destination)])
                        self.assertEqual(caught.exception.code, 2)
                        self.assertIn(message, stderr.getvalue())
                        self.assertNotIn('Traceback', stderr.getvalue())
                        for spy in spies:
                            spy.assert_not_called()
            self.assertEqual(existing.read_text(), 'preserve this result\n')
            self.assertTrue(dangling.is_symlink())
            self.assertFalse((root / 'absent-target').exists())

    def test_main_reports_filesystem_error_as_cli_error(self):
        with patch.object(cli, 'command', side_effect=PermissionError('fixture output denied')), \
             contextlib.redirect_stderr(io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as caught:
                cli.main(['demo', '--offline'])
        self.assertEqual(caught.exception.code, 2)
        self.assertIn('fixture output denied', stderr.getvalue())
        self.assertNotIn('Traceback', stderr.getvalue())

    def test_write_still_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'prior.jsonl'
            output.write_text('preserve this result\n')
            with self.assertRaises(FileExistsError):
                cli.write_lines(output, [{'replacement': True}])
            self.assertEqual(output.read_text(), 'preserve this result\n')


if __name__ == '__main__':
    unittest.main()
