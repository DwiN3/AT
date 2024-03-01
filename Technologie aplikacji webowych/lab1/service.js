function isEven(tab) {
    return tab.filter(number => number % 2 != 0);
}

module.exports = {
    isEven: isEven
}