-- Skończone

module Main where

data Tree a = EmptyNode 
              | Leaf a 
              | Node a (Tree a) (Tree a)

drzewo_wysokosc :: Tree a -> Int
drzewo_wysokosc EmptyNode = 0
drzewo_wysokosc (Leaf _) = 1
drzewo_wysokosc (Node _ lewo prawo) = 1 + max (drzewo_wysokosc lewo) (drzewo_wysokosc prawo)

ilosc_lisci :: Tree a -> Int
ilosc_lisci EmptyNode = 0
ilosc_lisci (Leaf _) = 1
ilosc_lisci (Node _ lewo prawo) = ilosc_lisci lewo + ilosc_lisci prawo

wyswietl_drzewo :: Show a => Tree a -> String
wyswietl_drzewo tree = wyswietl_drzewo_pomoc tree 0
  where
    wyswietl_drzewo_pomoc EmptyNode _ = ""
    wyswietl_drzewo_pomoc (Leaf a) indent = replicate indent ' ' ++ show a ++ "\n"
    wyswietl_drzewo_pomoc (Node a lewo prawo) indent =
      wyswietl_drzewo_pomoc prawo (indent + 4) ++
      replicate indent ' ' ++ show a ++ "\n" ++
      wyswietl_drzewo_pomoc lewo (indent + 4)

dodanie_elementu :: Ord a => a -> Tree a -> Tree a
dodanie_elementu x EmptyNode = Leaf x
dodanie_elementu x (Leaf a)
  | x < a = Node a (Leaf x) EmptyNode
  | otherwise = Node a EmptyNode (Leaf x)
dodanie_elementu x (Node a lewo prawo)
  | x < a = Node a (dodanie_elementu x lewo) prawo
  | otherwise = Node a lewo (dodanie_elementu x prawo)

drzewo_do_listy :: Tree a -> [a]
drzewo_do_listy EmptyNode = []
drzewo_do_listy (Leaf a) = [a]
drzewo_do_listy (Node a lewo prawo) = drzewo_do_listy lewo ++ [a] ++ drzewo_do_listy prawo

lista_do_drzewa :: Ord a => [a] -> Tree a
lista_do_drzewa [] = EmptyNode
lista_do_drzewa [x] = Leaf x
lista_do_drzewa xs =
  let (lewo, srodek : prawo) = splitAt (length xs `div` 2) xs
  in Node srodek (lista_do_drzewa lewo) (lista_do_drzewa prawo)

main = do
  let mkTree  = Node 5 (Node 3 EmptyNode EmptyNode) (Node 7 EmptyNode EmptyNode) 
  putStrLn("Drzewo binarne: \n")
  putStrLn (wyswietl_drzewo mkTree)