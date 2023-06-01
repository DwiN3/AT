-- Skończone

module Main where

sumRekurencyjna :: Num a => [a] -> a
sumRekurencyjna [] = 0
sumRekurencyjna [_] = 0
sumRekurencyjna (_:x:xs) = x + sumRekurencyjna xs

sumFold :: Num a => [a] -> a
sumFold xs = foldl (\acc (i, x) -> if odd i then acc + x else acc) 0 (zip [0..] xs)


main = do
  let lista = [1,3,7,3,99,4]
  putStrLn $ "Lista:  " ++ show lista
  putStrLn $ "\nSuma elementów o nieparzystych indeksach (Rekurencyjna):  " ++ show (sumRekurencyjna lista)
  putStrLn $ "Suma elementów o nieparzystych indeksach (foldl):         " ++ show (sumFold lista)