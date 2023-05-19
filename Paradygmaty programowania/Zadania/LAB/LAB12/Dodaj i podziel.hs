-- Skończone

module Main where

dodaj_i_podziel :: Fractional a => [a] -> [a]
dodaj_i_podziel = map (\x -> (x+1)/2)

dodaj_i_podziel_a :: Fractional a => [a] -> [a]
dodaj_i_podziel_a = map ((/2) . (+1))

dodaj_i_podziel_b :: Fractional a => [a] -> [a]
dodaj_i_podziel_b = (map(/2)) . (map(+1))


main = do
  let lista = [1, 4, 8, 16, 32]
  
  putStrLn $ "Funkcja dodaj_i_podziel:\n" ++ show (dodaj_i_podziel lista);
  putStrLn $ "\nFunkcja dodaj_i_podziel_a:\n" ++ show (dodaj_i_podziel_a lista);
  putStrLn $ "\nFunkcja dodaj_i_podziel_b:\n" ++ show (dodaj_i_podziel_b lista);