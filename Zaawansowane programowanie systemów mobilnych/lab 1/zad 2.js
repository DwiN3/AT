const _ = require('lodash');

function sum(){
    const result = _.sum(arguments);
    console.log(result);
}

sum(1,2,3,4,5);
sum(2,4,6);