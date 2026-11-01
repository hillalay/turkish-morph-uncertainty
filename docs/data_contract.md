Veri Sözleşmesi
Bu belge, morfoloji modülünün ürettiği deney örnekleri ile LLM modülünün
ürettiği olasılık sonuçları arasındaki sözleşmedir. Veri kümesi v1 olarak kalır;
sonuçlar v1 veya aşağıda tanımlanan v2 olabilir. Normatif makine şemaları
data/schema/dataset.schema.json, data/schema/model_result.schema.json ve
data/schema/model_result.v2.schema.json dosyalarıdır. Dosyalar UTF-8, her satırda
tek JSON nesnesi bulunan JSONL biçimindedir ve boş satırlar yok sayılır.
Deney örneği
Her kayıt aşağıdaki alanları taşır:
Alan	Anlamı
sample_id	Kalıcı, proje genelindeki yayımlanmış örnekler arasında benzersiz kimlik. Yayımlandıktan sonra içerik değişirse eski kimlik yeniden kullanılmaz.
prefix	Modele hedeften önce aynen verilecek metin; sondaki boşluklar anlamlıdır.
target	Olasılığı ölçülecek, boş olmayan metin.
lemma	Deneyde seçilen çözümlemenin lemması.
pos	Seçilen çözümlemenin Universal Dependencies tarzı büyük harfli sözcük türü.
morphological_features	Seçilen çözümlemenin sabit özellik nesnesi.
affix_count	Seçilen çözümlemedeki affixes dizisinin uzunluğu.
morphological_analyses	Bir veya daha fazla olası çözümleme.
selected_analysis_index	Bağlam/deney için seçilen çözümlemenin sıfır tabanlı indeksi.
scope	Kaydın ilk deney kapsamına uygunluğunu açıkça bildiren deney türü ve dışlama bayrakları.


morphological_features her zaman tense, mood, polarity, person,
number ve case anahtarlarını içerir. Uygulanmayan veya bilinmeyen kategori
null olur; anahtar atılmaz. İzin verilen değerler JSON Schema'da listelenir.
Üst düzey lemma, pos, morphological_features ve affix_count, seçili
çözümlemenin denormalize edilmiş görünümüdür ve doğrulayıcı bunların eşitliğini
kontrol eder.

Doğrulayıcı tek JSONL dosyası içindeki sample_id tekrarlarını yakalar. Birden
fazla veri dosyası yayımlanacaksa proje-geneli benzersizliği korumak için bu
dosyaların birleşimi de aynı doğrulayıcıdan geçirilmelidir. Sonuç kayıtları
yalnızca doğrulanan veri kümesinde bulunan kimliklere bağlanabilir.

Bir çözümleyici birden fazla çözüm döndürürse hiçbiri üstüne yazılmaz:
çözümler morphological_analyses içinde sabit bir sırayla saklanır ve deneyde
kullanılan çözüm selected_analysis_index ile belirtilir. Otomatik seçim
güvenilir değilse veri üretimi tamamlanmış sayılmaz; seçim insan incelemesine
veya ileride tanımlanacak açık bir seçim politikasına bırakılır.
İlk deney kapsamı
Sürüm 1 veri kümesi yalnızca tek sözcüklü hedefleri ve aşağıdaki iki deney
türünü kabul eder:
- simple_verb_inflection: en az bir açık çekim eki taşıyan, tense, mood,
  polarity, person ve number değerleri belirlenmiş basit fiil çekimi;
- noun_case: en az bir açık durum eki taşıyan isim veya özel isim. Yalın durum
  bu gruba dahil değildir.
Türetim zincirleri, ek-fiil, klitikler ve çok sözcüklü hedefler bu sürümde
kapsam dışıdır. Bunlar için şimdiden ayrıntılı segmentasyon kuralları koymak,
ilk deneyin ihtiyaç duymadığı dilbilimsel tercihleri veri sözleşmesine kalıcı
olarak bağlar. Daha güvenli yaklaşım, temel çekim ve durum eki ölçümleri
sabitlenene kadar bu yapıları reddetmektir.

Her kayıt scope.experiment_type yanında has_derivation, has_copula,
has_clitic ve is_multiword_target alanlarını açıkça false olarak taşır.
Şema bu bayraklardan biri true ise kaydı reddeder; doğrulayıcı ayrıca hedefte
boşluk ve seçili çözümlemede türetim eki bulunmasını denetler. Bu bayraklar veri
üreticisinin açık beyanıdır: kapsam belirleme tam bir morfolojik çözümleyici
olmadan yalnızca metinden güvenilir biçimde çıkarılamayacağı için veri üretim
aşamasında da kontrol edilmelidir. Bayraklar seçili çözümlemeyi niteler;
seçilmemiş alternatif analizler kaybedilmeden saklanabilir.
Metin birleştirme ve hedef aralığı
Model girdisi yalnızca prefix + target ile oluşturulur. Araya boşluk,
normalizasyon karakteri veya başka bir ayraç eklenmez; hiçbir karakter
silinmez. Gerekli ayırıcı boşluk prefix içinde bulunmalıdır. Her iki metin de
Unicode NFC olmalıdır.

Karakter indisleri Python/JSON dizgilerindeki Unicode kod noktalarına göre,
sıfır tabanlı ve yarı açık [başlangıç, bitiş) aralığıdır:
target_char_span = [len(prefix), len(prefix) + len(target)]
Örneğin prefix="Dün ", target="geldim" için model girdisi Dün geldim,
hedef aralığı [4, 10) olur. Tokenizer farklı bir ofset birimi (ör. UTF-8
baytı) veriyorsa LLM modülü bunu önce Unicode kod noktası aralığına
çevirmelidir.
Ek sayımı
Sürüm 1 için kural şudur: seçili çözümlemede lemma sınırından sonra gelen,
yüzeyde açıkça gerçekleşmiş her bağlı çekim biçimbirimi bir ek sayılır. Bir
allomorf tek ek sayılır; aynı yüzey parçası birden fazla özellik kodlasa bile
(portmanto ek) tek segment sayılır; sıfır biçimbirim ve kökteki ses değişimi
sayılmaz. Kaynaştırma sesi ayrı ek değildir ve ilişkili olduğu ekin yüzey
biçimine dahil edilir. Her sayılan birim affixes dizisinde yüzey biçimi,
etiketi ve inflectional türüyle, yüzey sırasına göre yer alır. Türetim ekleri
ve enklitikler bu sürümde reddedilir. Dolayısıyla
affix_count == len(selected_analysis.affixes) olmalıdır.

Dilbilim ekibinin veri kümesini büyütmeden önce kesinleştirmesi gereken kararlar:
- türetim zincirlerinde lemma sınırının nerede kurulacağı;
- ek-fiil ve birleşik zaman biçimlerinin tek mi birden çok mu sayılacağı;
- soru parçacığı, de/da ve benzeri klitiklerin kapsamı;
- yardımcı fiilli ve çok sözcüklü hedeflerin desteklenip desteklenmeyeceği;
- çözümleyiciler farklı segmentasyon verdiğinde kullanılacak öncelik kuralı.
Bu kararlardan biri mevcut sayıları değiştirirse sözleşme/veri kümesi sürümü
artırılmalı; yayımlanmış sample_id kayıtları yerinde değiştirilmemelidir.
Model sonucu — v1 (geriye dönük sözleşme)
Her sonuçta sample_id, model_id, model_revision,
evaluation_config_id, token_count, token_logprobs, word_nll ve
token_normalized_nll bulunur. Bir sonucun benzersiz anahtarı:
(sample_id, model_id, model_revision, evaluation_config_id)
model_revision değişmez ağırlık/snapshot revizyonunu,
evaluation_config_id ise tokenizer, şablon ve çıkarım ayarlarının sürümlü
kimliğini belirtir. Başarılı kayıtta token_logprobs yalnızca hedefi örten
tokenların doğal logaritmalı koşullu olasılıklarıdır;
log p(t_i | prefix, t_1...t_{i-1}) değerlerini metin sırasında taşır:
token_count = len(token_logprobs)
word_nll = -sum(token_logprobs)
token_normalized_nll = word_nll / token_count
word_nll hedef dizisinin toplam negatif log olabilirliğidir.
token_normalized_nll modelin kendi tokenlarına göre ortalamadır; tokenizerları
farklı modeller arasında doğrudan aynı ölçekmiş gibi yorumlanmamalıdır.

alignment.target_char_span yukarıdaki türetilmiş aralıkla aynı olmalıdır.
target_token_indices model girdisindeki ardışık token indislerini,
target_token_char_spans ise aynı tokenların Unicode karakter aralıklarını
taşır. Bu aralıklar boşluksuz biçimde hedefin tamamını ve yalnızca hedefi
örtmelidir.

Tokenizerın hedefi bir veya birden çok tokena bölmesi geçerlidir; token sayısı
ve iç bölünme sabit değildir. Ancak ilk veya son token hedef karakter sınırının
dışına taşıyorsa ya da token ofsetlerinde boşluk/örtüşme varsa bu sürümde hedef
olasılığı güvenilir biçimde ayrıştırılmış sayılmaz. Bazı başında-boşluk kullanan
tokenizerlar bu nedenle hata üretebilir; değerlendirme raporunda model başına
hizalama hata oranı ayrıca verilmelidir. Bu katı politika eksik sonuç üretmeyi,
fark edilmeden yanlış sözcük olasılığı hesaplamaya tercih eder.

Tokenizer hedef sınırlarını tam olarak hizalayamıyorsa veya güvenilir ofset
üretemiyorsa metrik hesaplanmaz. Kayıt status="alignment_error" ve
error.code="TARGET_ALIGNMENT_FAILED" ile yazılır; token ve NLL alanları ile
alignment alanı null olur. Böylece hata görünür kalır ve yanlış değerler
sessizce analize girmez.

## Model sonucu v2 — Token Boundary Policy v2

### Pilot bulguları ve kapsam

Nafie-473M (`nafie-ai/nafie-473M`) tokenizer'ıyla mevcut beş pilot örnekte,
v1'in yalnızca sözcüğü kapsayan katı sınır politikası **0/5 PASS** verdi:
beş örneğin tamamında `alignment_error` oluştu. Bağlamın sonundaki tek ASCII
boşluğunu ölçülen hedefe taşıyan alternatif deney **5/5 PASS** verdi. Bunlar
Issue #2'de bildirilen tokenizer pilotu sonuçlarıdır. 1 Kasım 2026'da yerel
önbellekten, ağ erişimi kapalı olarak yeniden doğrulandı: katı aralık kontrolü
**0/5**, yeni `check_alignment` kapısı **5/5 PASS** verdi. Kullanılan snapshot
`20477766abe4ca9e5ef3ef8891164a346887bba8`, Transformers `5.22.1`, Tokenizers
`0.23.3`; yeni politikanın config kimliği
`leading_space_in_target_v1:sha256:21cfab8126872fe8943219a33ce78423c929ccaf77bfd4fc14588d816636d194`
oldu. Model ağırlığı yüklenmedi. İlk pilotun revizyonu ayrıca saklanmadığı için
bu bilgiler yeni doğrulama koşulunu tanımlar. Beş örnek diğer modeller veya tüm
Türkçe girdiler için başarı garantisi değildir.

Ceren'in dataset şeması, beş pilot kayıt, `prefix` ve `target` değişmez.
Yeni ölçüm politikası yalnızca sonuç sözleşmesine uygulanır; v1 şeması yerinde
korunur. Eski sonuçlar yeni politika kullanılmış gibi yeniden etiketlenmez.

### `leading_space_in_target_v1` tanımı

`prefix` sonundaki boşluk dizisi tam olarak bir U+0020 olmalıdır. Sonda boşluk
bulunmaması, iki veya daha fazla boşluk, tab/NBSP veya bunları izleyen ASCII
boşluk bu politika için desteklenmez. Hedef tek, boş olmayan NFC sözcüktür.

```python
context = prefix[:-1]
measured_target = " " + target
full_text = context + measured_target  # Aynen prefix + target.
linguistic_target_char_span = [len(prefix), len(full_text)]
measured_target_char_span = [len(context), len(full_text)]
```

İki aralık da Unicode kod noktalarıyla sıfır tabanlı, yarı açıktır. Örneğin
`prefix="Dün "`, `target="geldim"` için dilbilimsel aralık `[4, 10)`, ölçülen
aralık `[3, 10)` olur. Morfolojik özellikler ve ek sayısı yalnızca `geldim`
sözcüğünü niteler; ölçülen token dizisi başındaki boşluğu da içerir.

### Boşluk dahil ölçümün matematiksel anlamı

`C = tokenize(context)` ve tam metnin tokenları `F = C + T` olmak üzere,
ölçülecek değer tokenizer'ın seçtiği ardışık hedef token yolu için
`P(T | C) = product_i P(T_i | C, T_<i)` olacaktır. Metin düzeyindeki kısa
gösterimi `P(" " + target | context)` şeklindedir. Bu, v1'deki
`P(target | prefix)` ile aynı koşullu olay değildir. Boşluk ile sözcük tek
tokenda birleşebildiği için ayrı bir boşluk tokenının olasılığını sonradan
çıkararak v1 değerine dönüştürmek genel olarak mümkün değildir. Bu skor,
aynı metni decode eden tüm alternatif token yollarının toplamı da değildir.

Alan adları korunur: `word_nll = -sum(token_logprobs)` artık boşluk dahil
**ölçülen dizinin** toplam NLL'sidir; yalnız dilbilimsel sözcüğün NLL'si diye
yorumlanmaz. `token_normalized_nll = word_nll / token_count` ve `token_count`
aynı ölçülen diziyi esas alır. Birim doğal logaritmayla nattır. Farklı ölçüm
politikaları veya tokenizerlarla elde edilen skorlar tek deney koşulu olarak
birleştirilmez.

### v1/v2 alanları ve sürüm seçimi

v1 dosyası ve kimliği `urn:turkish-llm-morphology:schema:model-result:v1`
değişmez. Eski, `schema_id` taşımayan kayıtlar v1 ile doğrulanır. v2'nin ayrı
şema kimliği `urn:turkish-llm-morphology:schema:model-result:v2` olup her v2
kaydı bunu `schema_id` alanında taşır. Bilinmeyen veya null kimlik reddedilir;
v2 alanları bulunan kimliksiz bir kayıt v1'in kapalı şemasından geçemez.
Aynı JSONL dosyasında iki sürüm bulunabilir; otomatik dönüşüm yapılmaz.

| Alan | v2 anlamı |
| --- | --- |
| `schema_id` | Yukarıdaki v2 kimliği; zorunlu sürüm ayıracı. |
| `measurement_policy` | Tam olarak `leading_space_in_target_v1`; hata kayıtlarında da zorunlu. |
| `tokenization_config` | Tokenizer kimliği, değişmez snapshot revizyonu, kütüphane sürümleri ve sabit ayarlar. |
| `evaluation_config_id` | Politika ve tokenizer ayarlarından türetilen doğrulanabilir kimlik. |
| `alignment.linguistic_target_char_span` | Orijinal hedef sözcüğün aralığı. |
| `alignment.measured_target_char_span` | Baştaki tek boşluk dahil ölçülen aralık. |
| `alignment.target_token_indices` | Özel token içermeyen tam girdide, ölçülen hedefin ardışık token konumları. |
| `alignment.target_token_char_spans` | Aynı tokenların ölçülen hedefi eksiksiz kapsayan karakter aralıkları. |

v1'in belirsiz `alignment.target_char_span` alanı yalnızca v1'de kalır; v2'de
iki açık aralık kullanılır. `sample_id`, `model_id`, `model_revision`,
`evaluation_config_id`, `token_count`, `token_logprobs`, `word_nll`,
`token_normalized_nll`, `status` ve `error` adları korunur. Benzersiz sonuç
anahtarı da değişmez. Başarılı kayıtta bütün metrikler ile hizalama zorunludur,
`error=null` olur. Başarısızlıkta `status="alignment_error"`,
`error.code="TARGET_ALIGNMENT_FAILED"` ve açıklayıcı `error.message` yazılır;
dört metrik alanı ve `alignment` null olur. Politika/config kimliği korunur.

### Ayarlar ve deney kimliği

`tokenization_config` şu alanların tamamını taşır:

- `tokenizer_id`, `tokenizer_revision`: Değişmez tokenizer dosyalarının kaynağı
  ve snapshot kimliği. Model ağırlık revizyonu ayrıca `model_revision` içindedir.
- `transformers_version`, `tokenizers_version`: Kullanılan kütüphane sürümleri.
- `use_fast=true`, `add_special_tokens=false`, `padding=false`, `truncation=false`,
  `skip_special_tokens=false`, `clean_up_tokenization_spaces=false`.

Chat template, metin normalizasyonu, elle BOS/EOS ekleme ve tokenizer
ayarlarını yüklemeden sonra değiştirme bu politikada kullanılmaz. Tokenizer'ın
dahili normalizer/pre-tokenizer ayarları değişmez snapshot içindeki dosyalardan
gelir; ek yükleme override'ları desteklenmez. Farklı ayar gerektiren modeller
için yeni sürümlü politika/config sözleşmesi gerekir.

Kimlik üretimi `turkish_llm_morphology.token_boundary.evaluation_config_id`
ile yapılır. `{"measurement_policy": ..., "tokenization_config": ...}` nesnesi
`json.dumps(ensure_ascii=False, sort_keys=True, separators=(",", ":"),
allow_nan=False)` ile serileştirilir; UTF-8 baytlarının SHA-256 özeti alınır.
Kimlik `leading_space_in_target_v1:sha256:<64 küçük hex karakter>` biçimindedir.
Doğrulayıcı hata kayıtlarında dahi özeti yeniden hesaplar. Politika veya
ayar değişip eski kimlik korunursa kayıt reddedilir. Aynı doğrulama girdisinde
v1/v2'nin aynı config kimliğini paylaşması da reddedilir. Ayrı dosyalardan
analiz yapılacaksa birleşimleri birlikte doğrulanmalıdır. Kimlik beyan edilen
ayarları bağlar; tokenizer'ın gerçekten o snapshot'tan geldiğini çevrimdışı
sonuç şeması tek başına kanıtlayamaz.

### Hizalama kapısı, BOS/EOS ve teacher forcing

`check_alignment` sadece tokenizer kullanır. Aşağıdaki koşulların tamamı
sağlanmadan model olasılığı hesaplanmamalıdır:

1. Desteklenen politika, tek ASCII boşluk ve NFC önkoşulları sağlanır.
2. `context` ve `full_text`, özel token, padding veya truncation olmadan
   tokenlaştırılır. `context_ids == full_ids[:len(context_ids)]` olmalıdır.
3. `m=len(context_ids)>0` olmalıdır. Hedef tokenları tam metnin `[m, n)`
   son ekidir; `target_token_indices = list(range(m, n))` olur.
4. Token ID/ofset sayıları eşleşir. Ofsetler geçerli Unicode kod noktalarıdır;
   bağlam tokenları ölçülen hedefe taşamaz. Hedefin tüm token aralıkları pozitif
   uzunluklu ve ardışıktır; ilk başlangıç ve son bitiş ölçülen aralıkla aynıdır.
   Arada boşluk veya örtüşme olamaz. Byte/UTF-16 ofsetleri bu uygulamada otomatik
   çevrilmez; ilgili tokenizer ayrıca doğrulanıp kod noktalarına uyarlanmalıdır.
5. Hem bağlamın hem tam metnin decode çıktısı, temizleme ve özel token atlama
   kapalıyken orijinal metinle birebir eşleşir.
6. BOS/EOS dahil `all_special_ids` içindeki hiçbir ID girdide bulunamaz.
   EOS hedef olasılığına dahil değildir. BOS kapalı olduğundan boş bağlam
   reddedilir; sessizce BOS eklenmez.

Nafie pilotunda causal LM teacher forcing girdisi aynen `full_ids` olur. Sıfır
tabanlı hedef tokenı `j`, `logits[0, j-1, :]` dağılımından ölçülür. Yani hedef
ID'leri `full_ids[m:n]`, karşılık gelen logit dilimi `logits[0, m-1:n-1, :]`
olacaktır. Bunlar batch/padding eklenmeden önceki konumlardır. Başka bir BOS,
şablon veya padding eklenirse bu eşleşme kullanılamaz; mevcut politikada bu
değişiklikler yasaktır. Logit okuma ve NLL hesabı `token_nll.py` içinde uygulanır;
FP16 logits normalizasyondan önce FP32'ye çevrilir. Uygulama ayrıntıları ve gerçek
beş örneğin sonuçları [token_nll.md](token_nll.md) belgesindedir. Normatif v1/v2
alanları ve şemaları değişmemiştir.

Koşullardan biri sağlanmazsa hizalama sonucu `alignment_error` olarak döner;
alternatif politikaya geçilmez ve desteklenmeyen örnekler sessizce atlanmaz.
JSON Schema/sonuç doğrulayıcısı aralıkları, kimliği ve mevcut NLL denklemlerini
denetler; gerçek token ID kararlılığı ve decode için üretimden önce bu çalışma
zamanı kontrolü ayrıca zorunludur.

### Gelecek modeller için doğrulama prosedürü

Önce testler model ağırlıkları ve gerçek tokenizer indirmeden çalıştırılır:

```bash
python -m unittest discover -s tests -v
python scripts/validate_dataset.py dataset data/pilot/pilot_dataset.jsonl
```

Ardından kurulu tokenizer ortamında değişmez revizyon seçilerek yalnız tokenizer
pilotu çalıştırılır; bu komut model ağırlığı yüklemez:

```bash
python scripts/check_boundary_policy.py --model-id nafie-ai/nafie-473M --revision <commit>
```

Betik tam 40 karakterli küçük hex commit kimliği ister; `main` gibi hareketli
revizyonları reddeder. Bu kimliği ve kütüphane sürümlerini config'e kaydeder;
her örnek için JSON tanı satırı ve sonunda PASS toplamı yazar. Bunlar NLL
sonuçları değildir, v2 sonuç dosyası olarak kullanılmaz. `explore_tokenizer.py`
ilk katı sınır pilotunun tarihsel keşif betiğidir; v2 ölçüm kapısı değildir.

Her yeni tokenizer/model revizyonunda beş kayıtla başlandıktan sonra 30–50
örneklik daha geniş pilotta başarılı/hatalı hizalama sayısı ve oranı, sınır
taşmaları, ek sayısı/hedef uzunluğu dağılımları raporlanmalıdır. Başarı yalnız
geçen alt küme için geçerlidir; seçici dışlanma analiz edilmelidir. Sonraki model
entegrasyonu, `status="ok"` kapısından sonra öğretmen zorlaması konumlarını
ayrıca doğrulamalı ve tamamlanan v2 kayıtlarını veri kümesiyle birlikte mevcut
sonuç doğrulayıcısından geçirmelidir.

**Uygulama durumu:** Tokenizer hizalaması, sürümlü sonuç doğrulaması ve Nafie için
teacher forcing ile `token_logprobs`/NLL hesabı mevcut. Model sabit revizyonda
CUDA/FP16 yüklendi; beş gerçek pilot sonucu v2 doğrulamasından geçti. Sayısal
profil ve çalışma hataları ayrı çalışma raporunda tutulur; v2 sonuç şeması
yalnız `alignment_error` durumunu desteklediğinden sayısal hatalar bu durumla
karıştırılmaz. Ayrıntılar [token_nll.md](token_nll.md) belgesindedir.
Predictive entropy hâlâ uygulanmadı. Transformers'ın tokenizer aşamasındaki
`nafie` model tipi uyarısı bu sabit ortamda model yükleme/ölçümünü engellemedi.

Doğrulama
Kurulum ve temel kontroller:
python -m pip install -e .
python scripts/validate_dataset.py dataset data/pilot/pilot_dataset.jsonl
python scripts/validate_dataset.py results path/to/results.jsonl --dataset data/pilot/pilot_dataset.jsonl
python -m unittest discover -s tests -v
JSON Schema alanları, türleri, kontrollü değerleri ve başarı/hata kayıt
şekillerini doğrular. Betik ayrıca sample_id ve bileşik sonuç anahtarı
benzersizliğini, NFC kullanımını, seçili çözümlemeyi, ek sayısını, hedef/token
aralıklarını ve NLL denklemlerini kontrol eder.
Planlanan araştırma genişletmeleri ve açık metodolojik kararlar
Bu bölüm araştırma yol haritasıdır; hizalama pilotunun güncel durumu 4. maddede
belirtilir. Diğer genişletmeler planlandı, henüz uygulanmadı. Bu bölüm yukarıdaki
normatif alanları ve geçerli kayıt koşullarını değiştirmez. Yeni genişletmeler
uygulanmadan önce deney protokolü ve uygun şema sürümü güncellenmelidir.

1. Predictive entropy — temel araştırma metriği (planlandı)
Token düzeyindeki predictive entropy, modelin hedef tokenı üretmeden hemen önce, sözlükteki tüm olası sonraki tokenlara dağıttığı olasılığın belirsizliğini ölçer:
\[
H_t = -\sum_{v \in V} p(v\mid c_t)\log p(v\mid c_t)
\]
Burada V modelin token sözlüğü, c_t ise hedef konumdaki önceki bağlam ile teacher forcing kapsamında verilmiş önceki hedef tokenlardır. Doğal logaritma kullanılır; birim nattır. token_logprobs içindeki hedefe ait tek olasılık entropy hesabı için yeterli değildir; her konumdaki tam logit/olasılık dağılımı gerekir.
- NLL, gerçekleşen hedef tokenın modele göre ne kadar düşük olasılıklı olduğunu ölçer.
- Predictive entropy, gerçekleşen hedef ne olursa olsun modelin tüm olası sonraki tokenlar arasında ne kadar kararsız olduğunu ölçer.
- Bu metrikler birbirinin yerine geçmez; düşük entropy yanlış fakat çok emin bir tahmine eşlik edebilir.
- Gelecekte token_entropies gibi konum bazlı değerler ve hedef kelime boyunca mean_entropy gibi önceden tanımlanmış özetler tutulabilir. Alan adları, null davranışı, BOS/EOS/padding dahil edilmesi ve hata durumları uygulama Issue'sunda kesinleştirilecektir.
- Farklı modellerin sözlük boyutu, token birimleri ve koşullandırma biçimi farklı olabilir. Bu nedenle ham entropy (ve token-normalized NLL) değerleri modeller arasında doğrudan eşdeğer ölçekler olarak yorumlanmayacaktır.
Durum: Planned / Not Implemented. Mevcut model_result.schema.json değişmez; entropy değerleri henüz zorunlu değildir.
2. Kontrollü karşılaştırma grupları (veri kümesi büyütülmeden önce karar)
Morfolojik karmaşıklık ile model belirsizliği arasındaki ilişkiyi yorumlarken kök, bağlam ve sözcük sıklığı gibi karıştırıcı etkenleri kontrol etmek için aşağıdaki aday alanlar değerlendirilecektir:
Planlanan alan	Açıklama
root_family_id	Aynı lemma/kök ailesinden oluşturulan gözlemleri ilişkilendirir.
context_template_id	Aynı bağlam şablonuna veya önceden belirlenmiş şablon ailesine dayanan gözlemleri ilişkilendirir.
comparison_group_id	Önceden belirlenmiş, karşılaştırılabilir koşulları içeren örnek kümesini tanımlar.


Karşılaştırmalar yalnızca ek sayısı farklı diye eşdeğer kabul edilmeyecek. Fiil zaman/kişi değişimlerinde bağlamların dilbilgisel ve anlamsal uygunluğu sağlanacak; mümkün olduğunca aynı kök, dengelenmiş bağlamlar ve yinelenen şablonlar kullanılacaktır. Bağlamlar farklıysa bu fark kaydedilip analizde denetlenecektir. Kelime uzunluğu, token sayısı ve uygun olduğunda sıklık da olası karıştırıcı değişkenlerdir.

Bu alanlar şu anki JSON şemasında yoktur. Ekleneceklerse şema ve doğrulayıcı birlikte sürümlendirilecek, veri üretimi başlamadan önce grup üyeliği kuralları belgelenecektir. Karşılaştırılabilirlik ve eşleştirme kalitesi yalnızca bir ID'nin bulunmasıyla garanti edilmez.

Durum: Dataset genişletilmeden önce tasarlanacak; Not Implemented.
3. Sıfır ekli kontrol grubu (veri kümesi büyütülmeden önce karar)
Mevcut sürüm 1 kapsamı simple_verb_inflection ve noun_case için en az bir açık çekim eki gerektirir; affix_count = 0 örnekleri şimdilik deney kapsamında geçerli değildir. Ancak “ek sayısı arttıkça belirsizlik artıyor mu?” sorusu için sıfır ekli karşılaştırma grubu yararlı olabilir.
- Yalın isimler (ev) ve çekimli isimler (evde) kontrollü bağlamlarda karşılaştırılabilir; ancak yalın biçimin ve ekli biçimin sözdizimsel rolleri farklı olduğundan doğrudan aynı bağlamı kullanmak her zaman geçerli değildir.
- Çıplak fiil kökü (gel) ile kişi/zaman çekimli fiil (geldim) aynı sözdizimsel işlevde olmayabilir; sıfır ekli fiili otomatik olarak eşdeğer kontrol saymayacağız.
- Sıfır ekli kayıtlar kabul edilecekse ayrı bir experiment_type veya açıkça tanımlanmış bir kontrol politikası, uygun bağlam şablonları ve gerekirse farklı karşılaştırma analizleri tasarlanacaktır.
- Sıfır ekli kontrol olmadan da çekimli biçimler arasında ek sayısı 1, 2, 3, ... düzeylerindeki ilişkiler test edilebilir; araştırmanın iddiası buna göre sınırlandırılmalıdır.
Durum: Değerlendirilecek; Not Implemented. Mevcut en-az-bir-ek kuralı değiştirilmemiştir.
4. Tokenizer hizalama politikası — ilk pilot tamamlandı
İlk beş örnekte katı politika 0/5, boşluk dahil alternatif 5/5 PASS verdi.
v1 korunur; v2'nin normatif kuralları ve 30–50 örneklik gelecek pilot prosedürü
yukarıdaki Token Boundary Policy v2 bölümündedir. Geniş pilot henüz yapılmadı.
5. Metrik ve geliştirme takvimi
Metrik / özellik	Amaç	Uygulama durumu
affix_count ve ek türleri	Morfolojik bağımsız değişkenler	Veri sözleşmesinde tanımlı; gerçek morfoloji modülü henüz yok
Token/subword sayısı	Tokenizer parçalanma etkisi	Hizalama kapısında hedef token konumları mevcut; model ölçümü henüz yok
word_nll	Hedef dizinin toplam negatif log olabilirliği	Nafie pilotunda teacher forcing ile uygulandı
token_normalized_nll	Model içi token başına ortalama NLL	Nafie pilotunda teacher forcing ile uygulandı
Token-level predictive entropy	Konum başına tüm sözlük dağılımının entropisi	Sonraki LLM Issue — planlandı
Hedef sözcük entropy özeti	Hedef token konumları boyunca ortalama vb.	Sonraki LLM Issue — planlandı
Perplexity	Uygun birim/normalizasyonla ortalama NLL'nin üstel dönüşümü	Gerekirse analiz aşamasında


Araştırma yorumu: Morfolojik ek sayısı, kelime uzunluğu, modelin subword sayısı ve bağlam uygunluğu birbirinden ayrı tutulacaktır. word_nll ve entropy farklı olguları ölçer. Modeller arası kıyas yapılırken farklı tokenizerların tanımladığı birimler ve geçerli örnek kümeleri raporlanacaktır.
6. Kararlar için tamamlanma koşulları
Veri kümesi pilotun ötesine büyütülmeden önce kontrollü karşılaştırma grupları ile sıfır ekli kontrol grubunun dahil edilip edilmeyeceği karara bağlanacak. Geniş tokenizer pilotundan sonra v2 hizalama politikasının kapsama oranı değerlendirilecek. Entropy alanlarının kesin şeması, hesaplama testleri ve çıktı biçimi ayrı bir geliştirme Issue'sunda belirlenecek. Her normatif değişiklik, ilgili JSON Schema / doğrulayıcı / testler ve belge birlikte güncellenerek sürümlendirilecektir.
