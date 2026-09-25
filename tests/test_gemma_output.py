from pathlib import Path
import unittest

import tensor_graphs

from main import decode_tokens, encode_text
from utils.decode import load_tokenizer


class GemmaOutputTests(unittest.TestCase):
    def testGemmaCompletesHiMyNameWithJasmine(self):
        project_root = Path(__file__).resolve().parents[1]
        model_path = project_root / "models" / "google" / "gemma-3-270m"
        tokenizer = load_tokenizer([str(model_path), "gemma-3-270m"])
        session = tensor_graphs.LLMSession(
            "gemma-3-270m",
            str(model_path),
            tensor_graphs.HeuristicSearchDelegate(),
            max_sequence_length=32,
            disable_caching=True
        )

        conversation_tokens = encode_text(tokenizer, "Hi, my name", is_first=True)
        expected_string = " is <strong>Jasmine"
        generated_string = ""

        while len(generated_string) <= len(expected_string):
            next_token = session.generate_step(conversation_tokens)
            self.assertNotEqual(next_token, -1)
            conversation_tokens.append(next_token)
            generated_string += decode_tokens(tokenizer, [next_token])
            self.assertTrue(
                generated_string.startswith(expected_string[: len(generated_string)]),
                f"Generated {generated_string!r}, expected prefix of {expected_string!r}",
            )


if __name__ == "__main__":
    unittest.main()
