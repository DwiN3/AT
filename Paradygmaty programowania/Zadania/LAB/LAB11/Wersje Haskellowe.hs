-- Skończone

module Main where

srednia_1 :: [Int] -> Int
srednia_1 list = sum list `div` length list

srednia_2 :: [Int] -> Int
srednia_2 list = foldr (+) 0 list `div` length list

splaszcz :: [[x]] -> [x]
splaszcz list = foldl (++) [] list

iloczyny_1 :: [Int] -> [Int]
iloczyny_1 list = map (\x -> x * length list) list

iloczyny_2 :: [Int] -> [Int]
iloczyny_2 list = map (* length list) list

main = do
  let list1 = [1, 2, 3, 4, 5, 6]
  putStrLn $ "Średnia sum i length:   " ++ show list1 ++ " = " ++ show (srednia_1 list1)
  putStrLn $ "Średnia foldr i length: " ++ show list1 ++ " = " ++ show (srednia_2 list1)

  let list2 = [[1,2,3], [5], [8,9]]
  putStrLn $ "Splaszcz:               " ++ show list2 ++ " = " ++ show (splaszcz list2)

  let list3 = [1,3,5]
  putStrLn $ "Iloczyny lambda:        " ++ show list3 ++ " = " ++ show (iloczyny_1 list3)
  putStrLn $ "Iloczyny operator *:    " ++ show list3 ++ " = " ++ show (iloczyny_2 list3)