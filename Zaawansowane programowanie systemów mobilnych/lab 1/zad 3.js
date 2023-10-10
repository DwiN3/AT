const _ = require('lodash');

function sum() {
    let result = 0;
    let allNumbers = true;

    for(let n = 0; n < arguments.length; n++) {
        if(typeof arguments[n] !== 'number') {
            console.log(`Błędny typ: Argument ${arguments[n]} (pozycja ${n + 1})`);
            allNumbers = false
        }
        else result += arguments[n];
    }
  
    if(allNumbers) console.log(result);
}
sum(1,2,3,4,5);
sum(1, 2, 'text', 4, 'string');
sum(2, {}, 6);
