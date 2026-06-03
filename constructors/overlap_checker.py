import json
import os
import re
from collections import defaultdict
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from config.parser import Parser
class ScalableFuzzyOverlapChecker:
    def __init__(self, language: str = 'hungarian', base_dataset_path: str = 'datasets', threshold: float = 0.85):
        self.conf = Parser()
        self.conf.get_args()

        self.language = language
        self.base_dir = os.path.join(
            os.path.expanduser('~'),
            self.conf.dataset_path,
            self.conf.language
        )
        self.threshold = threshold

        self.target_utterances = []  # Our "Gold Standard" test sets

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def load_manifest(self, dataset_name: str, is_test_set: bool):
        source_utterance = []

        dataset_path = os.path.join(self.base_dir, dataset_name, 'manifests')
        for split_file in os.listdir(dataset_path):
            manifest_path = Path(os.path.join(self.base_dir, dataset_name, 'manifests', split_file))

            print(f"Loading {dataset_name} ({split_file})...")
            count = 0

            with open(manifest_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    raw_text = data.get('text', '')
                    clean = self.clean_text(raw_text)

                    if len(clean) > 15:
                        entry = {
                            'dataset': dataset_name,
                            'filepath': data.get('audio_filepath', ''),
                            'clean_text': clean,
                            'raw_text': raw_text
                        }
                        if is_test_set:
                            self.target_utterances.append(entry)
                        else:
                            source_utterance.append(entry)

                        count += 1

            print(f"  -> Loaded {count} valid utterances.")

        if not is_test_set:
            return source_utterance

    def compare(self, dataset_name: str, chunk_size: int = 10000):
        print(f"\nBuilding TF-IDF Vectorizer for Character N-Grams...")
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4))

        source_utterances = self.load_manifest(dataset_name, is_test_set=False)
        all_texts = [u['clean_text'] for u in self.target_utterances] + [u['clean_text'] for u in source_utterances]
        vectorizer.fit(all_texts)

        target_matrix = vectorizer.transform([u['clean_text'] for u in self.target_utterances])
        source_matrix = vectorizer.transform([u['clean_text'] for u in source_utterances])

        print(f"\nCommencing High-Speed Fuzzy Search (Threshold: {self.threshold * 100}%)...")

        source_len = len(source_matrix)
        overlaps_found = []

        for start_idx in range(0, source_len, chunk_size):
            chunk_matrix = source_matrix[start_idx : start_idx + chunk_size]

            similarity_sparse = target_matrix.dot(chunk_matrix.T)
            similarity_coo = similarity_sparse.tocoo()
            mask = similarity_coo.data >= self.threshold

            source_indices = similarity_coo.row[mask] + start_idx
            target_indices = similarity_coo.col[mask]
            scores = similarity_coo.data[mask]

            overlaps_found.extend([
                {
                    'similarity_score': round(float(score), 3),
                    'test_leak': self.target_utterances[trg_idx],
                    'source_leak': source_utterances[src_idx]
                }
                for src_idx, trg_idx, score in zip(source_indices, target_indices, scores)
            ])

            overlaps_found.sort(key=lambda x: x['similarity_score'], reverse=True)
            print(f"\nTotal Overlaps Detected: {len(overlaps_found)}")

            output_path = os.path.join(self.base_dir, "overlaps.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(overlaps_found, f, indent=4, ensure_ascii=False)

            print(f"Results successfully saved to: {output_path}")
            return overlaps_found


        return overlaps_found

if __name__ == '__main__':
    checker = ScalableFuzzyOverlapChecker(language='hungarian', threshold=0.85)
    checker.load_manifest(dataset_name='fleurs', is_test_set=True)
    checker.load_manifest(dataset_name='common_voice', is_test_set=True)
    checker.load_manifest(dataset_name='voxpopuli', is_test_set=True)

    overlaps = checker.compare('yodas')

