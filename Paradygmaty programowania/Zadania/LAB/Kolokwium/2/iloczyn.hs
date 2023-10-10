productEvenIndices :: [Int] -> Int
productEvenIndices [] = 1
productEvenIndices [x] = x
productEvenIndices (x:_:xs) = x * productEvenIndices xs

productEvenIndicesTailRec :: [Int] -> Int
productEvenIndicesTailRec = helper 1
  where
    helper acc [] = acc
    helper acc [_] = acc
    helper acc (x:_:xs) = helper (acc * x) xs

productEvenIndicesFold :: [Int] -> Int
productEvenIndicesFold = foldl (\acc (i, x) -> if even i then acc * x else acc) 1 . zip [0..]


main :: IO ()
main = do
  let list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  
  putStrLn "Zwykła funkcja rekurencyjna:"
  print (productEvenIndices list) 
  
  putStrLn "Funkcja rekurencyjna z rekurencją ogonową:"
  print (productEvenIndicesTailRec list) 
  
  putStrLn "Funkcja wykorzystująca foldl:"
  print (productEvenIndicesFold list)
