# Turkish Morph Uncertainty

## Veri sözleşmesi

Morfoloji ve LLM modülleri arasındaki ortak veri biçimi
[`docs/data_contract.md`](docs/data_contract.md) içinde tanımlanır. Yeni veri kümeleri ve
model sonuçları, ilgili JSON Schema dosyalarıyla birlikte `scripts/validate_dataset.py`
kullanılarak doğrulanmalıdır.

Kurulum ve test:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

```bash
python scripts/validate_dataset.py dataset data/pilot/pilot_dataset.jsonl
python scripts/validate_dataset.py results path/to/results.jsonl --dataset data/pilot/pilot_dataset.jsonl
```
