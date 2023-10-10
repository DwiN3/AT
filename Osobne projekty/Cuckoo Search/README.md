# Cuckoo Search
 
## Wprowadzane dane do programu:

* populationSize - zwiększenie wartości tej zmiennej zwiększy liczbę osobników (kukułek) w populacji, co może zwiększyć szansę na znalezienie lepszego minimum globalnego. Jednakże zwiększenie tej wartości wiąże się również z dłuższym czasem obliczeń i większym zużyciem pamięci.

* probability - zmiana wartości tej zmiennej wpłynie na prawdopodobieństwo, z jakim kukułki będą przeprowadzać operację Lévy'ego. Zwiększenie tej wartości zwiększy prawdopodobieństwo wykonania operacji Lévy'ego, co z kolei może prowadzić do większej dywersyfikacji populacji i zwiększenia szansy na znalezienie lepszego minimum globalnego.

* alpha - wartość tej zmiennej określa stopień wpływu nowego rozwiązania na populację. Im większa wartość tej zmiennej, tym większy wpływ na populację będzie miało nowe rozwiązanie. Zwiększenie tej wartości może prowadzić do szybszej zbieżności algorytmu, ale jednocześnie zwiększa ryzyko zatrzymania się w minimach lokalnych.

* maxIterations - zwiększenie tej wartości zwiększy liczbę iteracji algorytmu, co zwiększy szansę na znalezienie lepszego minimum globalnego. Jednakże zwiększenie tej wartości również zwiększy czas obliczeń i zużycie pamięci.

* lb_l, lb_r, ub_l, ub_r - zmiana wartości granicznych dla zmiennych niezależnych wpłynie na zakres poszukiwań rozwiązania. Zwiększenie tych wartości zwiększy zakres poszukiwań, ale może również prowadzić do większego zużycia pamięci i czasu obliczeń.