const _ = require('lodash');
const Calculator = require('./Calculator.js')

const item1 = new Calculator([1,2,3,4,5])
const item2 = new Calculator([1,'text',3,{}])

item1.sum()
item1.subtract()

item2.sum()
item2.subtract()
