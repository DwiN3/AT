function sumPublicationYears(books) {
    var sum = 0;
    for (var _i = 0, books_1 = books; _i < books_1.length; _i++) {
        var book = books_1[_i];
        sum += book.publicationYear;
    }
    return sum;
}
var Books = [
    { title: "Metro 2033", author: "Dmitrij Głuchowski", publicationYear: 2015 },
    { title: "Pan Tadeusz", author: "Adam Mickiewicz", publicationYear: 1834 },
    { title: "Akademia pana Kleksa", author: "Jan Brzechwa", publicationYear: 1946 }
];
var sumYears = sumPublicationYears(Books);
console.log("Suma lat publikacji wszystkich książek:", sumYears);
