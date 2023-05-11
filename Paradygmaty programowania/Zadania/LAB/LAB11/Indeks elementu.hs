-- Skończone 

module Main where

index_last :: Eq a => a -> [a] -> Int
index_last x list = go x list (-1) 0
  where
    go _ [] res _ = res
    go y (z:zs) res idx = if y == z then go y zs idx (idx+1) else go y zs res (idx+1)

index_first :: Eq a => a -> [a] -> Int
index_first x list = go x list (-1) 0
  where
    go _ [] res _ = res
    go y (z:zs) res idx
      | res /= -1 = res
      | y == z = go y zs idx (idx+1)
      | otherwise = go y zs res (idx+1)

main = do
  let list = [1, 2, 3, 4, 2, 5, 2]
  let elem = 2
  putStrLn $ "\nLista: " ++ show list
  putStrLn $ "Indeks ostatniego wystąpienia elementu "++ show elem ++ ": " ++ show (index_last elem list)
  putStrLn $ "Indeks pierwszego wystąpienia elementu "++ show elem ++ ": " ++ show (index_first elem list)