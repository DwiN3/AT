class Product {
    constructor( id, name, category, image, price, description,sizes) {
      this.id = id;
      this.category = category;
      this.name = name;
      this.description = description;
      this.price = price;
      this.image = image;
      this.sizes = sizes;
    }
  }

const sizes = [40,41,42,44]

const ProductsList = [
    new Product(1, "Air Jordan 1", "Buty męskie", require('../../../../assets/PresentationApp/sneakersImage/Nike_Air_Jordan_1.png'), 899.99, 'Air Jordan 1 - ikona nie tylko sportowej mody, lecz również kultury ulicznej. Te legendarnie stylowe buty, zaprojektowane przez Michaela Jordana i Petera Moore\'a w 1985 roku, podbiły serca fanów na całym świecie swoim charakterystycznym designem oraz doskonałym komfortem noszenia. Wykonane z wysokiej jakości materiałów, Air Jordan 1 to nie tylko obuwie, to symbol niepowtarzalnego stylu i nieustającej pasji do sportu. Dostępne w różnorodnych kolorach i edycjach, te buty są niezrównane pod względem jakości i prestiżu. Zainwestuj w legendę i dołącz do historii ulicznej mody z Air Jordan 1.',sizes),
    new Product(2, "Nike Air Huarache", "Buty męskie", require('../../../../assets/PresentationApp/sneakersImage/Nike_Air_Huarache.png'), 629.99, 'Buty Nike Air Huarache - to doskonały model do chodzenia na co dzień stworzony dla wygody Twoich stóp. Detale z miękkiej skóry na cholewce łączą się z niezwykle przewiewnym połyskliwym materiałem przypominającym neopren, nadając efektowny styl. Niski kołnierz i konstrukcja przypominająca skarpetę wewnętrzną nadają smukły wygląd. Kultowy uchwyt na zapiętku i brak znaków markowych nadają butom styl wczesnych lat 90., który kochasz.',sizes),
    new Product(3, "Nike Air Max", "Buty męskie", require('../../../../assets/PresentationApp/sneakersImage/Nike_Air_Max.png'), 749.99, 'Nie ma nic wygodniejszego i bardziej niezawodnego – model Nike Air Max 90 pozostaje wierny swoim korzeniom dzięki kultowej podeszwie z waflowym bieżnikiem, przyszytym powłokom i klasycznym akcentom z tworzywa TPU. Nowe detale nadają całości nowoczesny wygląd, a amortyzacja Max Air zwiększa wygodę podczas biegu.',sizes),
    new Product(4, "Nike Air Trainer 5", "Buty męskie", require('../../../../assets/PresentationApp/sneakersImage/Nike_Air_Max_Alpha_Trainer_5.png'), 399.99, 'Ćwicz z mocą, która oszołomi całą siłownię, w butach Nike Air Max Alpha Trainer 5. Amortyzacja Max Air odpowiada za wygodę i stabilność podczas podnoszenia ciężarów, niezależnie od tego, czy ćwiczysz podczas lekkiego, czy wyczerpującego dnia. Szeroka płaska podstawa zapewnia zwiększoną stabilność i przyczepność podczas wszelkiego rodzaju intensywnych treningów, a dodatkowo sprawia, że prezentujesz się stylowo podczas kolejnych ćwiczeń.',sizes),
    new Product(5, "Nike Calm", "Buty męskie", require('../../../../assets/PresentationApp/sneakersImage/Nike_Calm.png'), 229.99, 'Wybierz spokój i wygodę bez względu na to, co robisz w ciągu dnia. Te wykonane z miękkiej, ale elastycznej, pianki lekkie klapki pasują do różnych stylizacji i łatwo je spakować. Podczas gdy wodoodporna konstrukcja sprawia, że te klapki idealnie nadają się na plażę lub nad basen, ich minimalistyczny wygląd sprawia, że można je nosić także w mieście. Czas włożyć je na stopy i ruszać przed siebie.',sizes),
    new Product(6, "Nike Dunk", "Buty męskie", require('../../../../assets/PresentationApp/sneakersImage/Nike_Dunk.png'), 629.99, 'Buty te zostały stworzone z myślą o parkiecie, ale zawładnęły ulicami. Kultowy koszykarski fason z lat 80. powraca w nowej odsłonie z klasycznymi detalami rodem z boiska. Wytrzymała skóra i tkanina rip-stop, detale o designie odbijającym światło i matowa niebieska podeszwa zewnętrzna tworzą solidną konstrukcję, która pomaga stawiać czoła trudnym warunkom. Niski, wyściełany kołnierz świetnie wygląda i daje wygodę w każdej sytuacji.',sizes),
    new Product(7, "Nike Mule", "Buty męskie", require('../../../../assets/PresentationApp/sneakersImage/Nike_Mule.png'), 529.99, 'Niezależnie od tego, czy zbliżasz się do 19. dołka, czy przygotowujesz się do kolejnej rundy, te zupełnie nowe buty Jordan 1 G Mule sprawią, że poczujesz się jak w raju. Są wygodne i łatwe w zakładaniu, rzucają się w oczy i prezentują ponadczasową estetykę Jordan.',sizes),
    new Product(8, "Nike Air Max Dn", "Buty męskie", require('../../../../assets/PresentationApp/sneakersImage/Nike_Air_Max_Dn.png'), 799.99, 'Poznaj technologię Air nowej generacji. Model Air Max Dn ma nasz system poduszek powietrznych Dynamic Air z rurkami zapewniającymi dwie różne wartości ciśnienia, który daje wrażenie sprężystości na każdym kroku. W efekcie fason wygląda futurystycznie i jest wystarczająco wygodny, aby sprawdzić się w ciągu dnia i wieczorem. Śmiało — Feel the Unreal.',sizes),
];

const CartList = [
  new Product(1, "Air Jordan 1", "Buty męskie", require('../../../../assets/PresentationApp/sneakersImage/Nike_Air_Jordan_1.png'), 899.99, 'Air Jordan 1 - ikona nie tylko sportowej mody, lecz również kultury ulicznej. Te legendarnie stylowe buty, zaprojektowane przez Michaela Jordana i Petera Moore\'a w 1985 roku, podbiły serca fanów na całym świecie swoim charakterystycznym designem oraz doskonałym komfortem noszenia. Wykonane z wysokiej jakości materiałów, Air Jordan 1 to nie tylko obuwie, to symbol niepowtarzalnego stylu i nieustającej pasji do sportu. Dostępne w różnorodnych kolorach i edycjach, te buty są niezrównane pod względem jakości i prestiżu. Zainwestuj w legendę i dołącz do historii ulicznej mody z Air Jordan 1.',sizes),
  new Product(2, "Nike Air Trainer 5", "Buty męskie", require('../../../../assets/PresentationApp/sneakersImage/Nike_Air_Max_Alpha_Trainer_5.png'), 399.99, 'Ćwicz z mocą, która oszołomi całą siłownię, w butach Nike Air Max Alpha Trainer 5. Amortyzacja Max Air odpowiada za wygodę i stabilność podczas podnoszenia ciężarów, niezależnie od tego, czy ćwiczysz podczas lekkiego, czy wyczerpującego dnia. Szeroka płaska podstawa zapewnia zwiększoną stabilność i przyczepność podczas wszelkiego rodzaju intensywnych treningów, a dodatkowo sprawia, że prezentujesz się stylowo podczas kolejnych ćwiczeń.',sizes),
  new Product(3, "Nike Air Max", "Buty męskie", require('../../../../assets/PresentationApp/sneakersImage/Nike_Air_Max.png'), 749.99, 'Nie ma nic wygodniejszego i bardziej niezawodnego – model Nike Air Max 90 pozostaje wierny swoim korzeniom dzięki kultowej podeszwie z waflowym bieżnikiem, przyszytym powłokom i klasycznym akcentom z tworzywa TPU. Nowe detale nadają całości nowoczesny wygląd, a amortyzacja Max Air zwiększa wygodę podczas biegu.',sizes),
  new Product(4, "Nike Air Huarache", "Buty męskie", require('../../../../assets/PresentationApp/sneakersImage/Nike_Air_Huarache.png'), 629.99, 'Buty Nike Air Huarache - to doskonały model do chodzenia na co dzień stworzony dla wygody Twoich stóp. Detale z miękkiej skóry na cholewce łączą się z niezwykle przewiewnym połyskliwym materiałem przypominającym neopren, nadając efektowny styl. Niski kołnierz i konstrukcja przypominająca skarpetę wewnętrzną nadają smukły wygląd. Kultowy uchwyt na zapiętku i brak znaków markowych nadają butom styl wczesnych lat 90., który kochasz.',sizes),
];

const ShopList = [
  { 
      id: 1,
      products: [
          new Product(1, "Nike Air Huarache", "Buty męskie", require('../../../../assets/PresentationApp/sneakersImage/Nike_Air_Huarache.png'), 629.99, '',sizes),
          new Product(2, "Nike Mule", "Kategoria", require('../../../../assets/PresentationApp/sneakersImage/Nike_Mule.png'), 529.99, '',sizes),
          new Product(2, "Nike Trainer 5", "Kategoria", require('../../../../assets/PresentationApp/sneakersImage/Nike_Air_Max_Alpha_Trainer_5.png'), 99.99, '',sizes)  
      ]
  },
  { 
      id: 2,
      products: [
          new Product(1, "Air Jordan 1", "Buty męskie", require('../../../../assets/PresentationApp/sneakersImage/Nike_Air_Jordan_1.png'), 899.99, '',sizes),
          new Product(2, "Nike Dunk", "Kategoria", require('../../../../assets/PresentationApp/sneakersImage/Nike_Dunk.png'), 629.99, '',sizes),
      ]
  },
];

export { Product, ProductsList, CartList, ShopList };