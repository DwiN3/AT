-- Skończone

module Main where

silnia_1 :: Int -> Int
silnia_1 i
  | i < 0 = -1
  | i == 0 = 1
  | otherwise = i * silnia_1 (i - 1)

silnia_2 :: Int -> Int
silnia_2 i = case i > 0 of
 True -> i * silnia_2 (i - 1)
 False -> if i < 0 then -1
                    else 1

main = do
  let a = 3
  let a_min = -3
  putStrLn $ "Silnia_1 z " ++ show a ++ " wynosi: " ++ show (silnia_1 a)
  putStrLn $ "Silnia_2 z " ++ show a ++ " wynosi: " ++ show (silnia_2 a)
  putStrLn $ "Silnia_1 z " ++ show a_min ++ " wynosi: " ++ show (silnia_1 a_min)
  putStrLn $ "Silnia_2 z " ++ show a_min ++ " wynosi: " ++ show (silnia_2 a_min)