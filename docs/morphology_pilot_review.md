# Morfoloji pilotu — insan inceleme listesi

Durum: **40 kayıt; 40 insan onaylı, 0 bekleyen.** Onaylar Hilal Ay tarafından açıkça bildirilen insan inceleme kararlarına dayanır; otomatik kontroller manuel doğrulama sayılmaz.

`morph-v1-001`–`morph-v1-040` kayıtlarının tamamı, kullanıcının açık insan incelemesi beyanıyla **Hilal Ay** adına onaylandı. Önceki geçici güncellememe kısıtları kullanıcı tarafından kaldırıldı. Bütün kayıtların manifest güncellemesinin gerçek UTC zamanı: `2026-11-24T13:03:29.727726+00:00`.

Onaydan önce ve sonra manifestteki 40 `sample_sha256` değeri canonical dataset üzerinden yeniden doğrulandı; bütün özetler korundu. Dataset, CSV, seed, ek sayımları ve ham analiz dosyaları değiştirilmedi. Toplam 79 alternatif analiz korunur. Onay yalnızca bağlama uygun seçilmiş analizin doğruluğunu ifade eder; diğer adayların silinmesine izin vermez. En az 30 manuel onay kabul kriteri karşılandı.

Bütün kayıtların `notes` alanı alternatiflerin korunması ve mevcut ek sayımlarının değiştirilmemesi açıklamasını taşır. Önceki 007 notu aynen korundu; 015, 016, 025, 026–028, 033 ve 034–040 için özel inceleme açıklamaları ilgili manifest kayıtlarına işlendi.

| Kayıt | Korunan teknik açıklama |
|---|---|
| 007 | `git → gid` ünsüz değişimi gövdeye aittir; ek sayılmaz. Lemma ve yüzey gövdesi ayrıdır. |
| 015 | `oku → ok` ünlü düşmesi ek sayılmaz. `Prog1: uyor` ve `A1pl: uz` ayrımı korunur. |
| 016 | `Fut: yacağ` gelecek zaman yüzeyi ve `A1pl: ız` kişi eki iki ayrı ek olarak korunur. |
| 025 | Dört özel isim/cins isim ve Acc/iyelik alternatifi korunur. Bağlamda cins isim + Acc seçilidir; belirsizliğin bağlamla çözülmesine örnektir. |
| 026–028 | Özel isim/cins isim alternatifleri korunur; bağlamdaki seçili analizler cins isimdir. |
| 033 | `çocuk → çocuğ` ünsüz yumuşaması ek sayılmaz; Accusative ve possessive alternatifleri korunur. |
| 034–036 | Sıfır türetimle isimleşmiş sıfat alternatifleri ham analizlerde korunur; seçili analiz cins isimdir. |
| 037–040 | Sıfır türetim alternatifleri korunur; seçili analizler cins isimdir. |

011–020 ek sayımları sırasıyla **2, 1, 2, 3, 2, 2, 2, 2, 3, 3** olarak korundu. Diğer kayıtların hâl özellikleri ve ek sayımları da değişmedi. `--require-reviewed` ve regresyon kontrollerinin sonuçları teknik dokümanda raporlanır.

**007 inceleme notu (aynen):**

> Kullanıcı incelemesinde onaylandı. git sözlük lemması ile gid yüzey biçimi arasındaki ünsüz değişimi belgelenmeli. Bu değişim ek olarak sayılmamalıdır. Lemma ve yüzey biçimi ayrı alanlarda korunmalıdır.

007 kaydında proje lemması `git`, ham analizde `dictionary_root="git"`, `dictionary_lemma="gitmek"`, `stem="gid"` ve `word="gidiyoruz"` ayrı alanlarda zaten korunuyor. `git` → `gid` ünsüz değişimi gövdeye aittir; ek sayılmaz. Seçili ekler `Prog1: iyor` ve `A1pl: uz`; `affix_count=2` aynı kaldı.

Her satırda doğal/anlamsal bağlamı, seçilen lemma/POS ve özellikleri, ek yüzeylerini ve alternatif çözümlemeleri inceleyin. Tam alternatifler `data/pilot/morphology_pilot_analyses.jsonl` dosyasındadır. Seçim, seed dosyasında açıkça yazılan Zeyrek morfem yolu ile yapılmıştır; otomatik bağlamsal ayrıştırma yapılmamıştır.

İnsan incelemesi sonrasında `data/pilot/morphology_pilot_review.json` içindeki ilgili kaydı `status="approved"`, gerçek `reviewer`, ISO `reviewed_at` ve gerekiyorsa `notes` ile güncelleyin. `sample_sha256` aynen korunmalıdır; içerik değişirse eski onay geçersizdir. İncelenmeyen veya çözülemeyen kayıtlar `pending` kalır. Bu liste inceleme yardımcısıdır; onayların kaynağı JSON manifestidir.

| ID | Prefix + target | Seçim | Yüzey segmentleri | Ek sayısı | Alternatif sayısı |
|---|---|---|---|---:|---:|
| morph-v1-001 | Dün akşam eve geldim | gel / VERB; tense=past, mood=indicative, polarity=positive, person=first, number=singular | gel+di+m | 2 | 1 |
| morph-v1-002 | Dün akşam eve geldik | gel / VERB; tense=past, mood=indicative, polarity=positive, person=first, number=plural | gel+di+k | 2 | 3 |
| morph-v1-003 | Dün akşam eve gelmedim | gel / VERB; tense=past, mood=indicative, polarity=negative, person=first, number=singular | gel+me+di+m | 3 | 1 |
| morph-v1-004 | Dün akşam eve gelmedik | gel / VERB; tense=past, mood=indicative, polarity=negative, person=first, number=plural | gel+me+di+k | 3 | 3 |
| morph-v1-005 | Dün erkenden okula gittim | git / VERB; tense=past, mood=indicative, polarity=positive, person=first, number=singular | git+ti+m | 2 | 1 |
| morph-v1-006 | Dün erkenden okula gitmedik | git / VERB; tense=past, mood=indicative, polarity=negative, person=first, number=plural | git+me+di+k | 3 | 3 |
| morph-v1-007 | Şimdi birlikte okula gidiyoruz | git / VERB; tense=progressive, mood=indicative, polarity=positive, person=first, number=plural | gid+iyor+uz | 2 | 1 |
| morph-v1-008 | Ali her sabah okula gider | git / VERB; tense=aorist, mood=indicative, polarity=positive, person=third, number=singular | gid+er | 1 | 4 |
| morph-v1-009 | Dün akşam uzun bir mektup yazdım | yaz / VERB; tense=past, mood=indicative, polarity=positive, person=first, number=singular | yaz+dı+m | 2 | 2 |
| morph-v1-010 | Dün akşam uzun bir mektup yazmadık | yaz / VERB; tense=past, mood=indicative, polarity=negative, person=first, number=plural | yaz+ma+dı+k | 3 | 3 |
| morph-v1-011 | Şimdi birlikte bir mektup yazıyoruz | yaz / VERB; tense=progressive, mood=indicative, polarity=positive, person=first, number=plural | yaz+ıyor+uz | 2 | 1 |
| morph-v1-012 | Ayşe her hafta bir mektup yazar | yaz / VERB; tense=aorist, mood=indicative, polarity=positive, person=third, number=singular | yaz+ar | 1 | 4 |
| morph-v1-013 | Dün akşam bu kitabı okudum | oku / VERB; tense=past, mood=indicative, polarity=positive, person=first, number=singular | oku+du+m | 2 | 1 |
| morph-v1-014 | Dün akşam bu kitabı okumadık | oku / VERB; tense=past, mood=indicative, polarity=negative, person=first, number=plural | oku+ma+dı+k | 3 | 3 |
| morph-v1-015 | Şimdi birlikte bu kitabı okuyoruz | oku / VERB; tense=progressive, mood=indicative, polarity=positive, person=first, number=plural | ok+uyor+uz | 2 | 1 |
| morph-v1-016 | Yarın birlikte bu kitabı okuyacağız | oku / VERB; tense=future, mood=indicative, polarity=positive, person=first, number=plural | oku+yacağ+ız | 2 | 1 |
| morph-v1-017 | Dün parkta arkadaşımı gördüm | gör / VERB; tense=past, mood=indicative, polarity=positive, person=first, number=singular | gör+dü+m | 2 | 2 |
| morph-v1-018 | Dün parkta arkadaşımızı gördük | gör / VERB; tense=past, mood=indicative, polarity=positive, person=first, number=plural | gör+dü+k | 2 | 3 |
| morph-v1-019 | Dün parkta arkadaşımı görmedim | gör / VERB; tense=past, mood=indicative, polarity=negative, person=first, number=singular | gör+me+di+m | 3 | 1 |
| morph-v1-020 | Dün parkta arkadaşımızı görmedik | gör / VERB; tense=past, mood=indicative, polarity=negative, person=first, number=plural | gör+me+di+k | 3 | 3 |
| morph-v1-021 | Dün akşam erkenden döndüm eve | ev / NOUN; number=singular, case=dative | ev+e | 1 | 1 |
| morph-v1-022 | Dün akşam birlikte yemek yedik evde | ev / NOUN; number=singular, case=locative | ev+de | 1 | 1 |
| morph-v1-023 | Bu sabah erkenden çıktım evden | ev / NOUN; number=singular, case=ablative | ev+den | 1 | 1 |
| morph-v1-024 | Taşınırken eşyaları çıkardık evlerimizden | ev / NOUN; number=plural, case=ablative | ev+ler+imiz+den | 3 | 1 |
| morph-v1-025 | Sonunda bitirdim bu uzun kitabı | kitap / NOUN; number=singular, case=accusative | kitab+ı | 1 | 4 |
| morph-v1-026 | Aradığım bilgiyi buldum bu kitapta | kitap / NOUN; number=singular, case=locative | kitap+ta | 1 | 2 |
| morph-v1-027 | Bu bilgiyi öğrendim aynı kitaptan | kitap / NOUN; number=singular, case=ablative | kitap+tan | 1 | 2 |
| morph-v1-028 | Yeni kapaklar hazırladık bu kitaplara | kitap / NOUN; number=plural, case=dative | kitap+lar+a | 2 | 2 |
| morph-v1-029 | Çocuklar her sabah yürüyerek gider okula | okul / NOUN; number=singular, case=dative | okul+a | 1 | 1 |
| morph-v1-030 | Öğrenciler bugün birlikte çalıştı okulda | okul / NOUN; number=singular, case=locative | okul+da | 1 | 1 |
| morph-v1-031 | Öğretmenler az önce ayrıldı okuldan | okul / NOUN; number=singular, case=ablative | okul+dan | 1 | 1 |
| morph-v1-032 | Yeni kitaplar gönderildi bütün okullara | okul / NOUN; number=plural, case=dative | okul+lar+a | 2 | 1 |
| morph-v1-033 | Bahçede oynarken gördüm bu çocuğu | çocuk / NOUN; number=singular, case=accusative | çocuğ+u | 1 | 2 |
| morph-v1-034 | Bu yeteneği fark ettim o çocukta | çocuk / NOUN; number=singular, case=locative | çocuk+ta | 1 | 3 |
| morph-v1-035 | Bu güzel haberi duydum o çocuktan | çocuk / NOUN; number=singular, case=ablative | çocuk+tan | 1 | 3 |
| morph-v1-036 | Renkli kalemler dağıttık bütün çocuklara | çocuk / NOUN; number=plural, case=dative | çocuk+lar+a | 2 | 3 |
| morph-v1-037 | Bu sabah dikkatle yıkadım arabayı | araba / NOUN; number=singular, case=accusative | araba+yı | 1 | 2 |
| morph-v1-038 | Anahtarları unuttum arabada | araba / NOUN; number=singular, case=locative | araba+da | 1 | 2 |
| morph-v1-039 | Eşyaları dikkatle indirdik arabadan | araba / NOUN; number=singular, case=ablative | araba+dan | 1 | 2 |
| morph-v1-040 | Yeni lastikler takıldı bütün arabalara | araba / NOUN; number=plural, case=dative | araba+lar+a | 2 | 2 |
