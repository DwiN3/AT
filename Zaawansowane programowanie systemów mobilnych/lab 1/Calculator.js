const _ = require('lodash');

module.exports = class Calculator{
  constructor(items){
    this.allNumbers = true;
    this.items = _.filter(items, item => {
      if (typeof item !== 'number') {
        console.log('Błędny typ:  Argument  ' + item);
        this.allNumbers = false;
        return false;
      }
      return true;
    });
  }

  sum(){
    if(this.allNumbers){
      const result = _.reduce(this.items, (acc, num) => acc + num, 0);
      console.log('Sum:', result);
    }
  }

  subtract(){
    if(this.allNumbers){
      const result = _.reduce(this.items, (acc, num) => acc - num);
      console.log('Sub:', result);
    }
  }
}