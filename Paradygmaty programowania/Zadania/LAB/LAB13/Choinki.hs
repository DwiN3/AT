-- Skończone

module Main where

fun_gałąź :: Int -> Int -> String -> String -> IO String
fun_gałąź h i linia znak
  | i == h = return linia
  | otherwise = do
    let nowa_linia = linia ++ znak ++ znak
    wynik <- fun_gałąź h (i - 1) nowa_linia znak
    return wynik

fun_odstęp :: Int -> Int -> String -> IO String
fun_odstęp h i linia
  | i == h = return linia
  | otherwise = do
    let nowa_linia = linia ++ " "
    wynik <- fun_odstęp h (i + 1) nowa_linia
    return wynik

fun_drzewo :: Int -> Int -> String -> IO ()
fun_drzewo h i znak
  | h == 0 = putStrLn ""
  | otherwise = do
    linia <- fun_gałąź h i znak znak
    spaces <- fun_odstęp (h - 1) 0 ""
    putStrLn (spaces ++ linia)
    fun_drzewo (h - 1) i znak

fun_wprowadź :: Int -> String -> IO ()
fun_wprowadź h znak = fun_drzewo h h znak

main = do
  print "Podaj rozmiar choinki"
  input_rozmiar <- getLine
  let rozmiar = read input_rozmiar :: Int
  if mod rozmiar 2 == 0
    then fun_wprowadź rozmiar "*"
    else fun_wprowadź rozmiar "#"
  if rozmiar /= 7
    then main
    else print "koniec"
