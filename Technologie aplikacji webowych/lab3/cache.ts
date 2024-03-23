type MyCache = {
    [key: string]: number;
  };
  
  function cachedSum(cache: MyCache, arg1: number, arg2: number): number {
    const key = `${arg1}_${arg2}`;
  
    if (cache[key] !== undefined) {
      console.log('\nPobrano z pamięci podręcznej:');
      return cache[key];
    }
  
    const result = arg1 + arg2;
  
    cache[key] = result;
    console.log('\nZapisano do pamięci podręcznej:');
  
    return result;
  }
  
  const cache: MyCache = {};
  console.log(cachedSum(cache, 1, 2));
  console.log(cachedSum(cache, 1, 2));
  console.log(cachedSum(cache, 3, 4));
  console.log(cachedSum(cache, 5, 4));