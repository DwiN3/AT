const _ = require('lodash');

const numbers = [1, -4, 83, 15, -6];
const min = _.min(numbers);
const max = _.max(numbers);
const mean = _.mean(numbers);

console.log('Minimalna wartość:', min);
console.log('Maksymalna wartość:', max);
console.log('Średnia arytmetyczna:', mean);