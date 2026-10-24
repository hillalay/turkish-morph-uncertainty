Veri Sözleşmesi
Bu belge, morfoloji modülünün ürettiği deney örnekleri ile LLM modülünün
ürettiği olasılık sonuçları arasındaki sürüm 1 sözleşmesidir. Normatif makine
şemaları data/schema/dataset.schema.json ve
data/schema/model_result.schema.json dosyalarıdır. Dosyalar UTF-8, her satırda
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
Model sonucu
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
Durum: Planlandı; sürüm 1'de uygulanmadı. Bu bölüm araştırma yol haritasıdır. Yukarıdaki normatif alanları, mevcut JSON şemalarını, doğrulayıcıyı veya geçerli kayıt koşullarını değiştirmez. Uygulamadan önce deney protokolü ve uygun şema sürümü güncellenmelidir.

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
4. Tokenizer hizalama politikası — pilotta sınanacak
Sürüm 1'in katı karakter-token sınırı kuralı korunur. Gerçek bir veya birkaç tokenizerla 30–50 örnekli pilot çalıştırılacak; her model/tokenizer için aşağıdakiler raporlanacaktır:
1. Toplam kayıt, başarılı hizalama sayısı ve alignment_error sayısı/oranı.
2. Başında boşluk içeren tokenların davranışı ve sınır taşması örnekleri.
3. Geçerli ve reddedilen örneklerin ek sayısı, hedef uzunluğu ve diğer özellikler bakımından dağılımı.
4. Olası alternatif hizalama politikalarının hedef dizinin hangi koşullu olasılığını ölçtüğüne etkisi.
Pilot görülmeden katı reddetme kuralı değiştirilmez. Hata oranı yüksekse alternatifler ayrı bir metodolojik karar ve sürüm değişikliği ile ele alınır. Amaç sessiz yanlış ölçümü önlemek kadar, bazı morfolojik grupların seçici biçimde analiz dışında kalmasını da saptamaktır.

Durum: Pilot doğrulaması bekleniyor.
5. Metrik ve geliştirme takvimi
Metrik / özellik	Amaç	Uygulama durumu
affix_count ve ek türleri	Morfolojik bağımsız değişkenler	Veri sözleşmesinde tanımlı; gerçek morfoloji modülü henüz yok
Token/subword sayısı	Tokenizer parçalanma etkisi	İlk LLM geliştirme aşaması — planlandı
word_nll	Hedef dizinin toplam negatif log olabilirliği	Sözleşmede tanımlı; hesaplama henüz uygulanmadı
token_normalized_nll	Model içi token başına ortalama NLL	Sözleşmede tanımlı; hesaplama henüz uygulanmadı
Token-level predictive entropy	Konum başına tüm sözlük dağılımının entropisi	Sonraki LLM Issue — planlandı
Hedef sözcük entropy özeti	Hedef token konumları boyunca ortalama vb.	Sonraki LLM Issue — planlandı
Perplexity	Uygun birim/normalizasyonla ortalama NLL'nin üstel dönüşümü	Gerekirse analiz aşamasında


Araştırma yorumu: Morfolojik ek sayısı, kelime uzunluğu, modelin subword sayısı ve bağlam uygunluğu birbirinden ayrı tutulacaktır. word_nll ve entropy farklı olguları ölçer. Modeller arası kıyas yapılırken farklı tokenizerların tanımladığı birimler ve geçerli örnek kümeleri raporlanacaktır.
6. Kararlar için tamamlanma koşulları
Veri kümesi pilotun ötesine büyütülmeden önce kontrollü karşılaştırma grupları ile sıfır ekli kontrol grubunun dahil edilip edilmeyeceği karara bağlanacak. İlk LLM pilotundan sonra hizalama politikası değerlendirilecek. Entropy alanlarının kesin şeması, hesaplama testleri ve çıktı biçimi ayrı bir geliştirme Issue'sunda belirlenecek. Her normatif değişiklik, ilgili JSON Schema / doğrulayıcı / testler ve belge birlikte güncellenerek sürümlendirilecektir.