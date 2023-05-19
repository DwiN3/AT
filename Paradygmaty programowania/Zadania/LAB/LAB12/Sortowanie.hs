-- Skończone

module Main where

wstaw :: Ord a => a -> [a] -> [a]
wstaw el [] = [el]
wstaw el (x:xs)
  | el <= x = el : x : xs
  | otherwise = x : wstaw el xs

sortuj_wstawianie :: Ord a => [a] -> [a]
sortuj_wstawianie [] = []
sortuj_wstawianie (x:xs) = wstaw x (sortuj_wstawianie xs)

sortuj_foldr :: Ord a => [a] -> [a]
sortuj_foldr = foldr wstaw []


main = do
  let lista = [19, 23, 5, 4, 3]
  
  putStrLn $ "Lista:\n" ++ show lista;
  putStrLn $ "\nFunkcja sortuj_wstawianie:\n" ++ show (sortuj_wstawianie lista);
  putStrLn $ "\nFunkcja sortuj_foldr::\n" ++ show (sortuj_foldr lista);