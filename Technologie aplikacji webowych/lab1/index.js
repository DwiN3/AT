let service = require('./service');

const tab = [1, 23, 42, 12, 11, 19, 21, 22, 18];
service.isEven(tab).forEach((e) => console.log(e));