function sum(){
    let result = 0;
    for(const el of arguments) result += el;
    console.log(result);
}

sum(1,2,3,4,5);
sum(2,4,6);