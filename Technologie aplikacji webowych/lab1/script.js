// ZADANIE 1
console.log('\n-----------------Zadanie 1----------------')
console.log('Pierwszy skrypt - działa');


// ZADANIE 2
console.log('\n-----------------Zadanie 2----------------');
var tab = [1,4,9,16,25,999];
console.log("Tablica:");
console.log(tab);

tab.pop();
console.log("\nDługość: "+tab.length);

console.log("\nTablica po modyfikacji:");
console.log(tab);

// ZADANIE 3
console.log('\n-----------------Zadanie 3----------------');
var tab = [2,1,3,5];

const addToTab = (tab, order, value) => {
    if(order == 0) {
        tab.unshift(value);
    } else {
        tab.push(value);
    }
    return tab;
}

console.log("Tablica");
console.log(tab);

console.log("\nTablice po modyfikacji:");
console.log(addToTab(tab,1,999));
console.log(addToTab(tab,0,-213));

// ZADANIE 4
console.log('\n-----------------Zadanie 4----------------');
let strNumbers = "1.2.3.4.5.6.7.8.9";
console.log(strNumbers);

const powStr = (str) => {
    let numbers = str.split('.');
    let powNumbers = numbers.map(number => Math.pow(parseInt(number), 2)).join('.');
    return powNumbers;
}

console.log(powStr(strNumbers));

// ZADANIE 5
console.log('\n-----------------Zadanie 5----------------');

const squareArea = (a) => {
    return a*a;
}

console.log('Pole kwadratu o boku 3:    '+squareArea(3));
console.log('Pole kwadratu o boku 12:   '+squareArea(12));

// ZADANIE 6
console.log('\n-----------------Zadanie 6----------------');
const students = ["Olek", "Janek", "Stefan", "Tymek", "Sławek"];

const randomPerson = (persons) => {
    let randomNumber = Math.floor(Math.random() * persons.length);
    return persons[randomNumber];
}
console.log("Wylosowana osoba to: "+ randomPerson(students));

// ZADANIE 7
console.log('\n-----------------Zadanie 7----------------');
const randomNumbers = (amount, delay) => {
    const generateNmbers = () => {
        amount--;
        console.log(Math.floor(Math.random() * 100));

        if(amount > 0){
            myVar = setTimeout(generateNmbers, delay);
        }
        else{
            clearTimeout(myVar);
        }
    }
    generateNmbers();
}

const welcomeUser = (name) => {    
    setInterval(() => console.log(`Witaj ${name}`), 3000);
}

randomNumbers(3, 1000);
welcomeUser('Roman');