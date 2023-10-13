const _ = require('lodash');
const user = require('./user');

const subject_w1 = _.find(user.allGrades, { weight: 1 });

if (subject_w1) console.log(`Przedmiot o wadze 1: ${subject_w1.subjectName}`);
else console.log('Nie znaleziono przedmiotu o wadze 1.');