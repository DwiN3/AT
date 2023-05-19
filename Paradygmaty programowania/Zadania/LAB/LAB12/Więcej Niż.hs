-- Skończone

module Main where
import Data.List (nub)

więcejNiż :: Eq a => Int -> [a] -> [a]
więcejNiż elem xs = [x | x <- nub xs, count x xs > elem]
  where
    count x = length . filter (== x)


main = do
  let lista = [1, 5, 2, 3, 2, 2 , 0, 5, 1, 2, 5]
  let pusta_lista = ([] :: [Int])
  
  putStrLn $ "Lista:" ++ show lista;
  
  putStrLn $ "\nElementy występujące więcej niż 1 razy:\nLista:       " ++ show (więcejNiż 1 lista) ++ "\nPusta lista: " ++ show  (więcejNiż 1 pusta_lista);
  
  putStrLn $ "\nElementy występujące więcej niż 2 razy:\nLista:       " ++ show (więcejNiż 2 lista) ++ "\nPusta lista: " ++ show  (więcejNiż 2 pusta_lista);
  
  putStrLn $ "\nElementy występujące więcej niż 3 razy:\nLista:       " ++ show (więcejNiż 3 lista) ++ "\nPusta lista: " ++ show  (więcejNiż 3 pusta_lista);