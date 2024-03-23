interface Book {
    title: string;
    author: string;
    publicationYear: number;
}

function sumPublicationYears(books: Book[]): number {
    let sum = 0;
    for (const book of books) {
        sum += book.publicationYear;
    }
    return sum;
}

const Books: Book[] = [
    { title: "Metro 2033", author: "Dmitrij Głuchowski", publicationYear: 2015 },
    { title: "Pan Tadeusz", author: "Adam Mickiewicz", publicationYear: 1834 },
    { title: "Akademia pana Kleksa", author: "Jan Brzechwa", publicationYear: 1946 }
];

const sumYears = sumPublicationYears(Books);
console.log("Suma lat publikacji wszystkich książek:", sumYears);