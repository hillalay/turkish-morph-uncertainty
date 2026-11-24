# Sprint 1 — Morfoloji ve pilot dataset

Branch: `feature/morphology-pilot`. Teknik doğrulama: 24 Kasım 2026.
40 taslak kayıt, 10 lemma × 4 biçim; 20 fiil ve 20 isim örneği.
23 kayıtta birden çok çözümleme, toplam 79 alternatif analiz korunur.
Ek sayısı dağılımı: 17 kayıtta 1, 15 kayıtta 2, 8 kayıtta 3 ek.
İnsan tarafından onaylanmış örnek sayısı **40**, bekleyen **0**.
`morph-v1-001`–`morph-v1-040` Hilal Ay tarafından açıkça incelenip onaylandı.
Onay bağlama uygun seçilmiş analize aittir; bütün alternatif analizler korunur.
Güncel insan kararları ve gerçek UTC güncelleme zamanı review JSON dosyasındadır.
Dataset içeriği ve kimliği korunmuştur; `morphology-pilot-v1-draft` sürüm etiketi değiştirilmemiştir.

## Kurulum

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[morphology]"
```

Mevcut `.venv` varsa yeniden oluşturmayın; yalnız ikinci komutu kullanın.
Doğrulanan ortam: Python 3.13.5, Zeyrek 0.1.3, NLTK 3.9.2.
Morfoloji bağımlılıkları `pyproject.toml` içinde isteğe bağlı extra olarak tanımlıdır;
temel veri sözleşmesi veya mevcut model paketleri için zorunlu hale getirilmedi.
Zeyrek Python wheel'i bu ortamda kurulup gerçek analizlerle çalıştırıldı.
Adapter yalnız tek kelime aldığı için cümle tokenizasyonu yapmaz ve NLTK
`punkt`/`punkt_tab` veri indirmesine ihtiyaç duymaz. İlk kurulumdan sonraki
analizler ağ erişimi gerektirmez.

## Analyzer ve gerçek Zeyrek nesneleri

```python
from turkish_llm_morphology.morphology import TurkishMorphAnalyzer
from turkish_llm_morphology.morphology.feature_extractor import extract_features
from turkish_llm_morphology.morphology.affix_counter import count_affixes

analyzer = TurkishMorphAnalyzer()
analyses = analyzer.analyze("geldim")
analysis = analyses[0]  # Bu kelimede tek analiz var; genel bir seçim politikası değildir.
print(analysis.lemma, analysis.dictionary_lemma, analysis.pos)
print([(m.id, m.surface, m.derivational) for m in analysis.morphemes])
print(extract_features(analysis))
print(count_affixes(analysis))
```

Gerçek çıktıdan:

```text
gel gelmek VERB
[('Verb', 'gel', False), ('Past', 'di', False), ('A1sg', 'm', False)]
{'tense': 'past', 'mood': 'indicative', 'polarity': 'positive',
 'person': 'first', 'number': 'singular', 'case': None,
 'aspect': None, 'possessive': None}
2
```

Adapter, public `Parse.formatted` metnini parçalamaz. Kurulu 0.1.3'ün
`MorphAnalyzer._parse()` metodundan gelen gerçek `SingleAnalysis` nesnelerindeki
`dict_item`, `stem`, `ending`, `(Morpheme, surface)` çiftleri ve
`group_boundaries` alanlarını okur. Public `Parse` ek yüzeylerini taşımadığı için
bu sürüme bağlı iç API kullanılır. Kaynaklar:
[Zeyrek projesi](https://github.com/obulat/zeyrek),
[PyPI 0.1.3](https://pypi.org/project/zeyrek/0.1.3/).
Uygulama kurulu wheel kaynağı üzerinden doğrulanmıştır; hareketli upstream
`main` branch'inin aynı nesne alanlarını koruduğu varsayılmaz.

Sözlükteki fiil lemması `gelmek`, v1 proje lemması sözlükteki `root` alanından
`gel` olur. Ses değişmiş gerçek gövde ayrıca saklanır: `gidiyorum` için lemma
`git`, gövde `gid`. İsimlerin sözlük lemması korunur; özel isimler `PROPN` olur.
Analizler sözlük kimliği/POS/morfem yolu üzerinden kararlı sıralanır;
hiçbir alternatif deduplicate edilmez veya kapsam dışı diye silinmez.

Zeyrek 0.1.3 arama sırasında gövde ve önbellekteki fonetik özellik kümelerini
değiştirebilir. Yeniden üretimde `geldik` sonrasında `geliyorum` analizinin
kaybolduğu gözlendi. Adapter, kelime başına aday gövde özelliklerini kopyalar,
`finally` ile geri yükler ve fonetik hesaplama önbelleğini temizler. Paylaşılan
grafiğe tekrar kenar eklenmemesi için backend bir kez yüklenir; bu işlemler
kilit altında yapılır. Kurulu Zeyrek dosyaları değiştirilmez. Kelime sırası ve
tekrarlanan wrapper oluşturma testleri bu davranışı doğrular.

## Feature extraction ve sözleşme sınırı

Özellikler son inflectional group üzerinden okunur. İçteki isim gruplarının
kişi/sayı özellikleri son fiil grubuna taşınmaz.

| Özellik | Temsil |
|---|---|
| Tense | Past, Pres, Prog1/Prog2, Aor, Fut, Narr etiketlerinden v1 değerleri |
| Mood | Imp, Cond, Neces, Opt; zaman etiketi bulunan yalın fiiller için indicative |
| Polarity | Fiilde Neg varsa negative, yoksa positive; isimde None |
| Person / Number | A1sg…A3pl; isimde sayı korunur, isim kişi alanı None |
| Case | Acc/Dat/Loc/Abl/Gen/Ins/Equ; isimde atlanan Nom geçişi nominative |
| Aspect | Prog1/Prog2 için progressive; diğer biçimlerde None |
| Possessive | P1sg…P3pl için ayrı person/number nesnesi; bulunmazsa None |

Zeyrek `parse_analysis`, Nom ve Pnon geçişlerini nesneden çıkarır; nominative
yorumu bu incelenmiş davranışa dayanır. Bulunmayan özellikler `None` kalır.
`Desr` etiketi v1 mood değerine zorla çevrilmez. `gelse` için tense/mood,
`gelmeliyim` için tense bilinmeyen/desteklenmeyen olarak None kalır; bu biçimler
pilotun zorunlu finite-verb özelliklerini karşılamadığından seçilemez.

**Şema değişikliği yapılmadı.** V1 yalnız `tense`, `mood`, `polarity`, `person`,
`number`, `case` alanlarını kabul eder. Aspect ve possessive Python API'sinde
ve ayrı ham analiz kanıtı dosyasında korunur. Bunları canonical kayda eklemek
şemayı bozar; ileride dataset analizinde gerekeceklerse sürümlü sözleşme
genişletmesi ayrıca kararlaştırılmalıdır. Ham morfemler de saklandığı için bu
bilgi kaybolmaz.

## Ek sayımı ve kapsam

Kök/gövdeyi temsil eden ilk morfem sayılmaz. Sonraki morfemlerde yüzeyi boş
olmayan her çekim segmenti bir ek sayılır. Sıfır A3sg sayılmaz; allomorf tek
segmenttir; kaynaştırma sesi ilgili ek yüzeyinin içindedir. Gövdedeki ötümlüleşme
ve diğer ses olayları sayılmaz. Etiketler özgün Zeyrek ID'leridir; örneğin Past.

`extract_affixes()` overt türetim ve çekim segmentlerini ayrı `type` ile
döndürür; `count_affixes()` yalnız çekim segmentlerini sayar. Örneğin:

- `geldim`: `gel+di+m`, 2.
- `gelmediler`: `gel+me+di+ler`, 3.
- `yazar` fiil analizi: `yaz+ar`, sıfır A3sg hariç 1.
- `kitaba`: `kitab+a`, kökteki p→b değişimi hariç 1.
- `evlerimizden`: `ev+ler+imiz+den`, 3; sahiplik, isim sayısından ayrı tutulur.

Seçili analizde türetim/sıfır dönüşüm, ek-fiil, birleşik zaman, klitik veya
overt ek yoksa `v1_scope()` hata verir. Seçilmemiş kapsam dışı alternatifler
aynen korunur. V1 pilot yalnız `simple_verb_inflection` ve `noun_case` içerir.

## Üretim ve doğrulama

```powershell
.venv/Scripts/python.exe scripts/build_dataset.py
.venv/Scripts/python.exe scripts/build_dataset.py --check
.venv/Scripts/python.exe scripts/validate_dataset.py dataset data/pilot/morphology_pilot_dataset.jsonl --morphology
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

İlk komut kontrollü seed kayıtlarını gerçek analizden üretir. Seed'deki lemma,
POS ve tam morfem ID yolu tek bir analizle eşleşmelidir; eşleşme yoksa/birden
çoksa işlem durur. Bu, açık bir taslak seçim tarifidir; otomatik bağlamsal
disambiguation veya insan onayı değildir. `--check` hiçbir çıktı yazmadan
seed → gerçek analiz → JSONL → CSV ve ham kanıt tutarlılığını denetler.

`validate_dataset.py --morphology`, mevcut ortak validator'a eklenen opt-in
üretici denetimidir. Eski komutların anlamı değişmez. Gerçek analizler yeniden
hesaplanır; yanlış lemma/ek yüzeyi/sayımı, kaybolan alternatif, yinelenen
ID veya prefix-target çifti, geçersiz seçili analiz ve boş/uygunsuz prefix
reddedilir. Genel şema kontrolleri de aynı validator üzerinden uygulanır.

Yerel Nafie tokenizer'ı ve mevcut Transformers ortamı varsa:

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
.venv/Scripts/python.exe scripts/build_dataset.py --check --check-nafie-alignment
.venv/Scripts/python.exe scripts/validate_dataset.py dataset data/pilot/morphology_pilot_dataset.jsonl --morphology --check-nafie-alignment
```

Bu kontrol model ağırlığı yüklemez, tokenizer indirmez. Revizyon
`20477766abe4ca9e5ef3ef8891164a346887bba8`; Transformers 5.22.1,
Tokenizers 0.23.3 ile 40/40 hizalama geçti. Prefix sonunda tam bir ASCII
boşluğu vardır; `prefix + target` aynen korunur. Morfoloji kurulumu tek başına
bu isteğe bağlı model ortamını kurmaz.

## Dosyalar ve CSV

| Dosya | İşlev |
|---|---|
| `data/pilot/morphology_pilot_seeds.jsonl` | 40 kontrollü bağlam ve açık analiz seçimi |
| `data/pilot/morphology_pilot_dataset.jsonl` | Ortak v1 şemasına uyan canonical taslak |
| `data/pilot/pilot_dataset.csv` | Aynı 40 kaydın kayıpsız CSV teslimi |
| `data/pilot/morphology_pilot_analyses.jsonl` | Tüm yapılandırılmış analizler, ek yüzeyleri, ek özellikler ve paket sürümleri |
| `data/pilot/morphology_pilot_review.json` | İçerik SHA-256'sına bağlı insan onay kayıtları; başlangıçta pending |
| `tests/fixtures/zeyrek_0_1_3.json` | 14 kelimeden yakalanmış gerçek analiz nesneleri; insan onayı değildir |

CSV'de `morphological_features`, `morphological_analyses`, `scope` hücreleri
UTF-8 JSON'dur. `affix_count` ve `selected_analysis_index` geri okunurken
integer'a çevrilir. Tırnak/virgül/Türkçe karakterler standart CSV escaping ile
korunur; prefix sonundaki boşluk silinmez. `read_csv()` ile okunan kayıtlar
canonical JSONL ile birebir karşılaştırılır. Yeni/çakışan schema yoktur.

Eski `data/pilot/pilot_dataset.jsonl`, sonuç dosyaları, Hilal'in yükleyicisi ve
inference kodu değiştirilmedi. Yeni örnekler `morph-v1-*` kimliklerini kullanır;
birleşik veri de ID benzersizliği kontrolünden geçer.

## İnsan incelemesi ve kabul kapısı

[40 satırlık liste](morphology_pilot_review.md) bağlam, seçim, yüzey segmentleri
ve alternatif sayısını gösterir. İnceleyen kişi hem bağlamdaki yorumu hem tüm
alternatifleri kontrol etmeli; çözülemeyen örneği onaylamamalıdır.
JSON manifestinde `status="approved"`, gerçek `reviewer`, ISO `reviewed_at`
ve gerekirse not girilir. İçerik özeti değiştirilmez. Veri değişince eski onay
geçersizleşir; builder eski manifesti görürse çıktı yazmadan durur.

```powershell
.venv/Scripts/python.exe scripts/build_dataset.py --check --require-reviewed
```

Bu komut en az 30 kayıt ve mevcut kayıtların tamamında geçerli insan onayı
ister. 40 insan onayı manifeste işlendi; kontrol **PASS** sonucuyla doğrulandı.
40/40 kayıt özeti eşleşti; insan incelemesi için dataset kabul kriteri karşılandı.

## Test sonuçları ve sınırlar

Başlangıç: 78/78 mevcut test geçti. Teslim: 106/106 test (28 yeni), atlanan yok.

Tam insan onaylarının manifeste uygulanması sonrası doğrulama
(`2026-11-24T13:03:29.727726+00:00` tarihli metadata güncellemesi):

- `scripts/build_dataset.py --check --require-reviewed`: **PASS**, 40/40 onay.
- Manifest/dataset `sample_sha256`: **40/40 eşleşme**, bütün eski özetler korundu.
- `python -B -m unittest discover -s tests -v`: **106/106 PASS**, 6,069 saniye; atlanan yok.
- Dataset, CSV, seed ve ham analiz dosyaları ile ek sayımları değişmedi.
- Bu güncelleme kullanıcı incelemesi için commit oluşturulmadan bırakıldı.
Gerçek Zeyrek testleri için morphology extra gerekir; yoksa integration
testleri açıkça skip olur, fixture tabanlı özellik testleri yine çalışır.
Yeni CSV I/O testlerinin ilk sandbox koşusunda iki PermissionError görüldü;
aynı testler değiştirilmeden izinli ortamda tüm paketle başarılı çalıştı.

Testler boş/hatalı/NFC dışı giriş, Türkçe karakterler, belirsizlik, türetim,
ek-fiil, ses olayı, yanlış lemma/ek sayısı/yüzeyi, kaybolan alternatifler,
tekrarlar, CSV round-trip, inceleme özeti ve analiz sırası yalıtımını kapsar.
Testlerdeki sentetik insan attestasyonları yalnız test belleğinde oluşturulur;
gerçek review manifestine yazılmaz.

Sınırlar: Tek alfabetik NFC kelime kabul edilir; apostrof, noktalama ve çok
kelimeli hedefler reddedilir. Zeyrek sözlüğü tüm doğru Türkçe biçimleri ve tüm
dilbilimsel çözümlemeleri garanti etmez; bilinmeyen kelime için boş liste döner.
Uzun/karmaşık türetim zincirleri pilot kapsamı ve testleri dışındadır. Açık
seçim tarifleri insan bağlam kontrolünün yerini almaz. Bağlamlar isimler için
doğal devrik söz dizimi kullanır; bütün kayıtların seçilmiş analizleri Hilal Ay tarafından onaylanmıştır.
Bu 40 kayıt, karmaşıklığın bağımsız etkisini kanıtlayan dengeli bir araştırma
tasarımı veya model karşılaştırması değildir.
