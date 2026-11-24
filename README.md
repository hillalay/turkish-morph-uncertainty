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

## Morfoloji pilotu — Sprint 1

Python 3.13.5 üzerinde Zeyrek 0.1.3 / NLTK 3.9.2 ile doğrulandı:

```powershell
.venv/Scripts/python.exe -m pip install -e ".[morphology]"
.venv/Scripts/python.exe scripts/build_dataset.py --check
.venv/Scripts/python.exe scripts/validate_dataset.py dataset data/pilot/morphology_pilot_dataset.jsonl --morphology
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Yeni canonical veri `data/pilot/morphology_pilot_dataset.jsonl`, kayıpsız CSV teslimi
`data/pilot/pilot_dataset.csv` dosyasıdır. **40 kayıt teknik kontrollerden geçti;
40 kayıt Hilal Ay tarafından insan incelemesinde onaylandı; bekleyen kayıt yok.** Mevcut beş örneklik `pilot_dataset.jsonl` ve
Hilal'in inference hattı aynı kalır. JSON Schema dosyaları değiştirilmedi.

[Kullanım, ek sayımı ve sınırlamalar](docs/morphology_pilot.md) ile
[40 örneklik insan inceleme listesi](docs/morphology_pilot_review.md) teslimin parçasıdır.
İnsan onayı `data/pilot/morphology_pilot_review.json` içinde kayıt içeriğinin SHA-256
özetiyle tutulur. `scripts/build_dataset.py --check --require-reviewed`, en az 30
kayıt bulunmasını ve mevcut kayıtların tamamının açık insan onayını zorunlu kılar;
40 onaylı kayıtla bu kontrol PASS sonucuyla doğrulanmıştır.
