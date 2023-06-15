module Main where
import Data.List

fun_1 :: String -> [String]
fun_1 str = filter (not . null) $ words str

fun_2 :: Char -> Double -> Double -> Double
fun_2 operator x y = case operator of
  '+' -> x + y
  '-' -> x - y
  '*' -> x * y
  '/' -> x / y
  _   -> error "Nieprawidłowy operator"

fun_3 :: [String] -> Double
fun_3 (operator:x:y:[]) = do
  let x_ = read x :: Double
  let y_ = read y :: Double
  fun_2 (head operator) x_ y_
fun_3 _ = error "Nieprawidłowe działanie"

main = do
  putStrLn "Podaj dane do działania (np. + 5 7):"
  input_znaki <- getLine
  let znaki = fun_1 input_znaki
  case length znaki of
    3 -> putStrLn $ "Wynik działania = " ++ show (fun_3 znaki)
    _ -> putStrLn "Błąd: Nieprawidłowe działanie"
