const _ = require('lodash');
const user = require('./user'); // Użyj względnej ścieżki do importu

const grades = user.allGrades.reduce((acc, subject) => acc.concat(subject.grades), []);
const mean = _.mean(grades);

console.log(`Imię: 	  ${user.name}`);
console.log(`Nazwisko: ${user.surname}`)
console.log('Średnia ważona:', mean);
