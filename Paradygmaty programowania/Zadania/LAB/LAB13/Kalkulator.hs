-- Skończone

module Main where
import Data.List

fun_1 :: String -> [String]
fun_1 = words

fun_2 :: Char -> Double -> Double -> Maybe Double
fun_2 operator x y = case operator of
  '+' -> Just (x + y)
  '-' -> Just (x - y)
  '*' -> Just (x * y)
  '/' -> if y == 0 then Nothing else Just (x / y)
  _   -> Nothing

fun_3 :: [String] -> Maybe Double
fun_3 (operator:x:y:[]) = do
  let x_ = read x :: Double
  let y_ = read y :: Double
  fun_2 (head operator) x_ y_
fun_3 _ = Nothing


main = do
  putStrLn "Podaj dane do działania (np. + 5 7):"
  input_znaki <- getLine
  let znaki = fun_1 input_znaki
  case fun_3 znaki of
    Just wynik -> putStrLn $ "Wynik działania = " ++ show wynik
    Nothing -> putStrLn "Błąd: Nieprawidłowe działanie"