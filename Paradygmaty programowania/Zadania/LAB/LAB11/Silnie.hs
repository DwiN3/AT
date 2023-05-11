-- Skończone

module Main where

silnia_1 :: Int -> Int
silnia_1 i
  | i < 0 = -1
  | i == 0 = 1
  | otherwise = i * silnia_1 (i - 1)

silnia_2 :: Int -> Int
silnia_2 i = case i of
  i
    | i < 0 -> -1
    | i == 0 -> 1
    | otherwise -> i * silnia_2 (i - 1)

main = do
  let a = 3
  let b = -15
  putStrLn $ "Silnia z " ++ show a ++ " wynosi: " ++ show (silnia_1 a)
  putStrLn $ "Silnia z " ++ show b ++ " wynosi: " ++ show (silnia_2 b)