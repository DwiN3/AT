-- Skończone

module Main where

data Tree a = EmptyNode | Leaf a | Node a (Tree a) (Tree a)

treeHeight :: Tree a -> Int
treeHeight EmptyNode = 0
treeHeight (Leaf _) = 1
treeHeight (Node _ left right) = 1 + max (treeHeight left) (treeHeight right)

leafCount :: Tree a -> Int
leafCount EmptyNode = 0
leafCount (Leaf _) = 1
leafCount (Node _ left right) = leafCount left + leafCount right

showTree :: Show a => Tree a -> Int -> String
showTree EmptyNode _ = "()"
showTree (Leaf a) level = replicate (3 * level) ' ' ++ show a
showTree (Node a left right) level =
  let space = replicate (3 * level) ' '
      subtree = showTree left (level + 1) ++ "\n" ++ space ++ show a ++ "\n" ++ showTree right (level + 1)
  in subtree

addElem :: Ord a => a -> Tree a -> Tree a
addElem a EmptyNode = Leaf a
addElem a (Leaf n)
  | a < n = Node n (Leaf a) EmptyNode
  | a > n = Node n EmptyNode (Leaf a)
  | otherwise = Leaf n
addElem a (Node n left right)
  | a < n = Node n (addElem a left) right
  | a > n = Node n left (addElem a right)
  | otherwise = Node n left right

treeToList :: Ord a => Tree a -> [a]
treeToList EmptyNode = []
treeToList (Leaf a) = [a]
treeToList (Node a left right) = treeToList left ++ [a] ++ treeToList right

listToTree :: Ord a => [a] -> Tree a
listToTree [] = EmptyNode
listToTree [x] = Leaf x
listToTree xs =
  let midIndex = length xs `div` 2
      leftList = take midIndex xs
      rightList = drop (midIndex + 1) xs
      midValue = xs !! midIndex
  in Node midValue (listToTree leftList) (listToTree rightList)

mkTree = Node 5 (Node 3 EmptyNode EmptyNode) (Node 7 EmptyNode EmptyNode)

main = do
  putStrLn "Wysokość drzewa:"
  print (treeHeight mkTree)

  putStrLn "\nLiczba liści:"
  print (leafCount mkTree)

  putStrLn "\nReprezentacja drzewa:"
  putStrLn (showTree mkTree 0)

  let updatedTree = addElem 4 mkTree
  putStrLn "\nDrzewo po dodaniu elementu 4:"
  putStrLn (showTree updatedTree 0)

  let treeList = treeToList updatedTree
  putStrLn "\nDrzewo skonwertowane do listy:"
  print treeList

  let newListToTree = listToTree treeList
  putStrLn "\nLista skonwertowana do drzewa:"
  putStrLn (showTree newListToTree 0)