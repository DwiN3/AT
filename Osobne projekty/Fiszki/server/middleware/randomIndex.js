module.exports = (arraySize) => {
    const array = [];
    if(arraySize > 15) arraySize = 15;
    while(array.length < arraySize){
        const random = Math.floor(Math.random() * arraySize) + 1;
        if(array.indexOf(random) === -1) array.push(random);
    }
    return array;
}