# Teacher forcing ve hedef token NLL

## Uygulama ve tekrar çalıştırma

`src/turkish_llm_morphology/token_nll.py`, Nafie pilotu için hedef token
log olasılıklarını ve NLL metriklerini hesaplar. `scripts/run_inference.py`
incelenmiş, SHA-256 ile doğrulanan yerel Nafie snapshot'ını bir kez yükler ve
beş örneği ayrı ayrı CUDA/FP16 üzerinde çalıştırır. Paket sürümleri değiştirilmez.

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
.venv/Scripts/python.exe scripts/run_inference.py
.venv/Scripts/python.exe scripts/validate_dataset.py results results/raw/nafie_473m_pilot_nll.v2.jsonl --dataset data/pilot/pilot_dataset.jsonl
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Ağırlıklar önbellekte bulunmalıdır; betik indirme veya CPU'ya geçiş yapmaz.
Kaynak incelemesi ve model yükleme ayrıntıları `nafie_forward_probe.md` içindedir.
Dataset, morfoloji modülü, v1/v2 şemaları ve doğrulayıcı değiştirilmedi.

## Ölçüm ve causal kaydırma

Politika `leading_space_in_target_v1` olarak sabittir:

```python
context = prefix[:-1]
measured_target = " " + target
full_text = prefix + target
```

`tokenize_aligned_target` mevcut hizalama kontrollerini çalıştırır ve bu
kontrollerden geçen tam token ID dizisini döndürür. Model girdisi bu ID'lerden
oluşturulur; ikinci bir tokenizasyon yapılmaz. Eski `check_alignment` API'si ve
döndürdüğü tanı nesnesi korunur.

Model her örnekte tek, padding içermeyen dizinin tamamını görür. Bütün gerçek
hedef tokenları girdidedir; `generate()` çağrılmaz. BOS/EOS, chat template ve
truncation kullanılmaz. Modelin maksimum bağlam uzunluğu aşılırsa hesaplama
reddedilir. Model `eval()` ve `torch.inference_mode()` altında çalışır.

Tam girdinin hedef son eki `[m, n)` ise:

```python
target_ids = input_ids[0, m:n]
target_logits = logits[0, m-1:n-1, :].float()
logprobs = torch.log_softmax(target_logits, dim=-1)
token_logprobs = logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
```

Dolayısıyla token konumu `i`, `i-1` konumunun dağılımından ölçülür. İkinci hedef
tokenın dağılımı birinci gerçek hedef tokenına da koşulludur. Tokenın kendi
konumundaki logits ile kendisine olasılık atanmaz; son tokenın logits'i kullanılmaz.

Normalizasyon her hedef konumu için **tüm sözlük** üzerinden yapılır; ardından
yalnız gerçek hedef ID'sinin değeri seçilir. `token_logprobs` metin sırasındadır.
Doğal logaritma kullanılır; birim nattır:

```python
word_nll = -sum(token_logprobs)
token_normalized_nll = word_nll / len(token_logprobs)
```

FP16 logits normalizasyondan **önce FP32'ye** çevrilir. Örneğin
`[65504, -65504]` için düşük olasılıklı tokenın log olasılığı FP16'da taşabilir;
FP32 normalizasyonu sonlu `-131008` değerini verir. Şekil, token ID sınırları,
tüm logits'in sonluluğu, FP32 dönüşümü ve seçilen log olasılıkları denetlenir.
Toplam ve ortalama, FP32 token log olasılıklarının Python float değerlerinden
hesaplanır. Bu dönüşüm, FP16 model forward'ında oluşmuş yuvarlama kaybını geri
almaz; tam FP32 modelle sayısal eşdeğerlik henüz ölçülmedi.

## Sonuç dosyaları ve hatalar

- `results/raw/nafie_473m_pilot_nll.v2.jsonl`: Her satır mevcut v2 şemasıyla ve
  `validate_result_records` ile doğrulanır. Dosyanın birleşimi de doğrulanır.
  UTF-8/LF baytları geçici dosyaya yazılır, sonra sonuç dosyası atomik değiştirilir.
- `results/raw/nafie_473m_pilot_nll_run.json`: Model/kod revizyonu, kütüphane
  sürümleri, GPU belleği, sayısal profil, hata listesi ve veri/sonuç dosyalarının
  SHA-256 özetleri bulunur. Bu çalışma raporu bir v2 sonuç kaydı değildir.

Hizalama başarısızsa model forward'ı çağrılmaz; v2'nin mevcut `alignment_error`
kaydı üretilir, dört metrik ve `alignment` null olur. Hesaplama veya sonuç
doğrulaması başarısızsa istisna yakalanır; örnek için başarılı v2 kaydı yazılmaz.
Çalışma raporunda `computation_error`, `NLL_COMPUTATION_FAILED`, orijinal hata
ve traceback saklanır; metrikler null olur. Sayısal hata yanlış biçimde
`alignment_error` olarak etiketlenmez. Herhangi bir örnek başarısızsa süreç
sıfırdan farklı kodla çıkar ve çalışma durumu `error` olur. Başarılı satırlar
bulunsa bile bu çalışma eksiksiz başarılı kabul edilmemelidir.

`evaluation_config_id`, mevcut v2 sözleşmesi gereği yalnız politika ve
`tokenization_config` özetidir; her sonuçta kayıtlıdır. Sayısal yürütme profili
`single_cuda_fp16_logsoftmax_fp32_v1` çalışma raporunda saklanır. Gelecekte farklı
model/normalizasyon hassasiyetleri karşılaştırılmadan önce bunları deney kimliğine
bağlayan sözleşme sürümlendirilmelidir. Şimdiki kimlik tek başına farklı sayısal
profilleri ayıramaz; sonuçlar bağlı çalışma raporuyla birlikte kullanılmalıdır.

## Gerçek pilot — 1 Kasım 2026

Model, tokenizer ve kod revizyonu:
`20477766abe4ca9e5ef3ef8891164a346887bba8`.
Python 3.13.5, PyTorch 2.14.1+cu126, Transformers 5.22.1, Tokenizers 0.23.3;
RTX 3060 Laptop GPU, CUDA 12.6. Çalışma çevrimdışı yapıldı.

| Örnek | Hedef | Token sayısı | Token log olasılıkları | word_nll | token_normalized_nll |
| --- | --- | ---: | --- | ---: | ---: |
| pilot-001 | geldim | 1 | [-8.45061684] | 8.45061684 | 8.45061684 |
| pilot-002 | gelmediler | 2 | [-7.37746143, -5.71973085] | 13.09719229 | 6.54859614 |
| pilot-003 | bahçede | 1 | [-9.20701790] | 9.20701790 | 9.20701790 |
| pilot-004 | yazar | 1 | [-6.97942448] | 6.97942448 | 6.97942448 |
| pilot-005 | koşuyoruz | 2 | [-7.79774952, -1.53341889] | 9.33116841 | 4.66558421 |

Tablo gösterim için yuvarlanmıştır; JSONL tam Python float değerlerini saklar.
Beş kayıt da `ok` ve v2 doğrulaması **5/5 PASS**; hizalama ve hesaplama hatası
sayısı sıfırdır. Önceki 61 test dahil **78/78 test** geçer; yeni testler sahte
logits ve CPU model/tokenizer örnekleriyle ağırlık indirmeden çalışır. PyTorch
kurulu değilse tensor testleri atlanır; çekirdek sözleşme testleri çalışmaya devam eder.

Bu skorlar boşluk dahil hedef token yolunun koşullu NLL'sidir. Yalnız kelimenin
boşluksuz olasılığı veya tüm olası token yollarının toplam olasılığı değildir.
Beş örnek bir uygulama pilotudur; morfolojik karmaşıklık hakkında istatistiksel
çıkarım sağlamaz. Farklı tokenizerların token-normalized NLL değerleri doğrudan
eşdeğer değildir. Padding/batch desteği, FP32 model karşılaştırması ve entropy
bu aşamada geliştirilmedi; özel Nafie mimarisi attention mask'i hâlâ kullanmaz.
