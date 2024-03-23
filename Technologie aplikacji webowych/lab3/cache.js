function cachedSum(cache, arg1, arg2) {
    var key = "".concat(arg1, "_").concat(arg2);
    if (cache[key] !== undefined) {
        console.log('\nPobrano z pamięci podręcznej:');
        return cache[key];
    }
    var result = arg1 + arg2;
    cache[key] = result;
    console.log('\nZapisano do pamięci podręcznej:');
    return result;
}
var cache = {};
console.log(cachedSum(cache, 1, 2));
console.log(cachedSum(cache, 1, 2));
console.log(cachedSum(cache, 3, 4));
console.log(cachedSum(cache, 5, 4));
