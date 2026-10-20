"""Örnek cevaplar için sabit prompt seti.

Her prompt: id, dil (tr/en), metin ve opsiyonel bir talimat kontrolü.
Kontroller basit ve otomatik: madde sayısı, cümle sayısı, kelime sınırı.
İngilizce promptlar "unutma"ya bakmak için: model Türkçe eğitimden sonra
İngilizce soruya hâlâ İngilizce cevap verebiliyor mu?
"""

PROMPTS = [
    # --- bilgi ---
    {"id": "tr_bilgi_1", "lang": "tr", "text": "Fotosentez nedir? Kısaca açıkla."},
    {"id": "tr_bilgi_2", "lang": "tr", "text": "Türkiye'nin en uzun nehri hangisidir ve hangi denize dökülür?"},
    {"id": "tr_bilgi_3", "lang": "tr", "text": "Mustafa Kemal Atatürk'ün hayatı hakkında kısa bir paragraf yaz."},
    {"id": "tr_bilgi_4", "lang": "tr", "text": "Enflasyon neden olur? Basit bir dille anlat."},
    {"id": "tr_bilgi_5", "lang": "tr", "text": "Bir bilgisayarın işlemcisi (CPU) ne işe yarar?"},
    # --- günlük / pratik ---
    {"id": "tr_pratik_1", "lang": "tr", "text": "Evde ekmek yapmak için basit bir tarif verir misin?"},
    {"id": "tr_pratik_2", "lang": "tr", "text": "Sınavdan önce stresimi nasıl azaltabilirim?"},
    {"id": "tr_pratik_3", "lang": "tr", "text": "Kışın ev bitkilerine nasıl bakılır?"},
    # --- yaratıcı ---
    {"id": "tr_yaratici_1", "lang": "tr", "text": "Sonbahar hakkında dört dizelik kısa bir şiir yaz."},
    {"id": "tr_yaratici_2", "lang": "tr", "text": "Kaybolan bir kedinin evine dönüşünü anlatan kısa bir hikaye yaz."},
    # --- dönüştürme ---
    {"id": "tr_ozet_1", "lang": "tr",
     "text": "Şu metni tek cümleyle özetle: Yapay zeka, bilgisayarların insan zekası gerektiren görevleri "
             "yerine getirmesini sağlayan bir bilim dalıdır. Görüntü tanıma, dil çevirisi ve oyun oynama gibi "
             "alanlarda büyük ilerlemeler kaydedilmiştir. Ancak etik ve güvenlik konularında hâlâ önemli "
             "sorular bulunmaktadır."},
    {"id": "tr_ceviri_1", "lang": "tr",
     "text": "Şu cümleyi İngilizceye çevir: Yarın sabah erkenden kütüphaneye gideceğim."},
    {"id": "tr_duzelt_1", "lang": "tr",
     "text": "Şu cümledeki yazım hatalarını düzelt: Herkez bu konuda birşey söylemek istiyordu ama kimse cesaret edemedi."},
    # --- talimata uyum (otomatik kontrol edilir) ---
    {"id": "tr_talimat_madde", "lang": "tr",
     "text": "Sağlıklı yaşam için tam olarak 3 öneri yaz. Her öneri ayrı bir madde olsun.",
     "check": {"type": "items", "n": 3}},
    {"id": "tr_talimat_cumle", "lang": "tr",
     "text": "Neden düzenli uyumalıyız? Sadece tek bir cümleyle cevap ver.",
     "check": {"type": "max_sentences", "n": 1}},
    {"id": "tr_talimat_kelime", "lang": "tr",
     "text": "İstanbul'u en fazla 20 kelimeyle anlat.",
     "check": {"type": "max_words", "n": 20}},
    {"id": "tr_talimat_madde_5", "lang": "tr",
     "text": "Türkiye'deki 5 büyük şehri madde madde listele, açıklama yazma.",
     "check": {"type": "items", "n": 5}},
    # --- akıl yürütme ---
    {"id": "tr_mantik_1", "lang": "tr",
     "text": "Ali'nin 3 elması var. Ayşe ona 4 elma daha veriyor, sonra Ali 2 elma yiyor. Ali'nin kaç elması kaldı?"},
    {"id": "tr_mantik_2", "lang": "tr",
     "text": "Bir tren saatte 80 km hızla 3 saat giderse kaç kilometre yol alır?"},
    # --- İngilizce (unutma / dil kayması) ---
    {"id": "en_1", "lang": "en", "text": "What is the capital of Australia?"},
    {"id": "en_2", "lang": "en", "text": "Explain in simple terms how a vaccine works."},
    {"id": "en_3", "lang": "en", "text": "Write a short poem about the ocean."},
    {"id": "en_4", "lang": "en", "text": "Give me three tips for learning a new language.",
     "check": {"type": "items", "n": 3}},
    {"id": "en_5", "lang": "en", "text": "If I have 10 apples and give away 4, how many do I have left?"},
]
