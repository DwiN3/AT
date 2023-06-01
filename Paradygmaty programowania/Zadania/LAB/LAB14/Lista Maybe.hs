-- Skończone

module Main where
import Control.Applicative
import Control.Monad
import Data.Foldable

justProduct_1 :: [Maybe Int] -> Maybe Int
justProduct_1 [] = Just 1
justProduct_1 (x:xs) = (*) <$> x <*> justProduct_1 xs

justProduct_2 :: [Maybe Int] -> Maybe Int
justProduct_2 [] = Just 1
justProduct_2 (x:xs) = do
  n <- x
  reszta <- justProduct_2 xs
  return (n * reszta)

justProduct_3 :: [Maybe Int] -> Maybe Int
justProduct_3 = foldl' (liftA2 (*)) (Just 1)


main = do
  let lista1 = [Just 1, Just 2, Just 3, Just 4]
  let lista2 = [Just 1, Just 2, Nothing, Just 4]
  print (justProduct_1 lista1)
  print (justProduct_2 lista1)
  print (justProduct_3 lista1)
  print (justProduct_1 lista2)
  print (justProduct_2 lista2)
  print (justProduct_3 lista2)