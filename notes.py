# satış tahmini için popüler bir yarışma olan m5 competition'dan yararlanılabilir ve fikir alınabilir.
# https://www.sciencedirect.com/science/article/pii/S0169207021001874

# Timeseries forecasting için darts kütüphanesini kullandım.
# Ancak Gbm için probleme regresyon problemi gibi de yaklaşılabilir.

# henüz validasyonları implemente etmedim.

# LGBM

## Benzer storelar incelenebilir Mesela satış hacmine göre / mean_item_per_income

## Item adları ile feature engineering yapılabilir. Burası için kesin bir şey öngörmek zor, itemların zaten halihazırda
## kateogrileri var.

## kategori bazında inceleme yapılabilir.
## Trx başına incomelarına ve countlarına baktım. Burada da benzer kategorileri modelleme fikri gibi bir şey doğuyor.

## lgbm için category counta göre bir modelleme yaptım. Bu kategorideki trx sayısına göre modelleme üzerinden doğan bir fikirdi.
## Darts içerisinde zaten lag değişkenlerini türetmek çok kolay ayrıca bunları da feature olarak besledim.

## Price değişkenini ben çok kullanmasam da anlamlı bilgiler içerecektir. Bu m5 içerisinde de ele alınan bir feature idi.
## Mesela price*amount üzerinde mağazalar segmente edilebilir. Bu mağazaların benzer özellikler taşıcacağı varsayılabilir.
## Ya da bunun için bir kümeleme algoritmasından yararlanılabilir. Direkt olarak storeları kümelemek gibi.

## Veri seti oldukça sparse zaman serilerinden oluşmakta bunlar drop edilip. Modellemeye devam edilebilir. Eşik değeri için ayrıca heuristic gerekecektir.

## günün sonunda LGBM için model 4 clusterla kuruldu. Her bir cluster için model kurmak bir seçenek olsa da verinin kartezyen çarpımı ile benim sistemim çok baş edemezdi.
## Tek bir model kurup bunun yerine cluster feature'ını model olarak besledim.

## Darts içerisindeki yapı ile bir çok zaman serisini tek seferde modellemek mümkün.
## Ne kadar çok data o kadar bilgi gibi düşünebiliriz. Bu sebeple ilk aşamada modeli hiç bir sparse serieyi drop etmeden kurdum.

# LSTM

## Hata metriği olarak custom bir metrik kullanmak çok daha doğru olacaktır.
## Bu metrik root(abs(y_real - ynaive)* price) / root(abs(y_real - y_lstm_pred)*price) olabilir.
## Eğer Hata 1'in üzerinde gelirse. Aslında naive forecastten ileri gidememiş oluruz.
## Eğer Hata 1'in altında gelirse naive forecastten daha iyi bir forecast yapmışızdır.
## Bu forecast hatası içerisinde priceın da etkisi saklıdır.

## lgbm içerisinde yapmış olduğum clustera karşılık(kategorilerin trx countu) Lstm içerisinde kullandığım cluster düzeni biraz daha farklı.
## Bu sefer itemların kategorilerde oluşturdukları satış ağırlıkları(miktar)
## model kategori bazında kuruldu bu ağırlıklara göre tahminler, item kırılımına yansıtılacak.

## Bu şekilde hem kartezyen çarpımından doğan ccomputainally expenseten kaçıldı. hem de tahminler kategori bazından alındıktan sonra satış ağırlıklarına göre
## itemlara kırıldı.

## Bu aşamada item-store ikilisinin kategori içeriisndeki ağırlığı da ayrıca hesaplanarak bir feature çıkartılabilir ve beslenebilirdi.

## Aslında bir item çok satıyorsa, bir store çok satıyorsa bir kategori çok satıyor bunların hepsi bir feature ve veya model kırılımı olabiilir.
## Bununla birlikte hibrit bir yapı kurulduğu için ben burada tüm clusterlar için tek bir model kurmuş olsam da, modeller clusterlar için ayrı ayrı kurularak
## hangi clusterda hangi model en az hata veriyor ise o modelin tahmini final_predicition olarak düşünülebilir.

## Bununla birlikte üç modelin sonucu da tek bir ensemble model olarak beslenebilir.

# Smoothing
## Deep learning ve ağaç tabanlı bir algoritmanın yanında tamamen farklı bir method kullanmanın iyi olabileceğini düşündüm.
## Ayrıca istatistiksel modelleri implemente etmesi de kolay.

## Smoothing kendi içerisine parametre almadan geçmiş sonuçlardan çok sapıtmayacak şekilde de tahminler üretebiliyor.
## Bu aşamda kartezyen çarpımının büyüklüğünü göz önünde bulundurarak datapre-process kısmında biraz daha fazla zaman harcadım.

## Burada yapılan işlemler. Eğer bir ürün son bir yılda satmıyorsa o ürünün satışının 0 basılması gibi adımları da kapsıyordu.
## scriptte ben son 1 yılda ve toplam veri setinin uzunluğu 0.5'ten kısa olanları eledim.
## ancak yeni bir ürün satışa sunulduğunda burada satış olmasına rağmen forecast yapılmaması gibi bir durum olacak.
## Bunun için son bir yıldaki satışlarına göre eleme yapıldıktan sonra, 0.5 koşulu ile sparse seriesler elenmezse,
## bir ürün satışa girdikten sonra modele sokulabilir.
## Mesela bir ürün son yılda 0 satıldıysa sıfır basarız, Eğer satışı varsa smoothing ile modellenebilir.


# Tüm modeller Streamlit üzerindeki bir apiye bağlandı.Prediction tetiklendikten sonra, database oluşturup connection ile basılması gerecektir.
# Ayrıca modelin ayda bir çalışacağını düşürsek bunun içinde schedule edilmesi için bir tool kullanılması gerecektir.

